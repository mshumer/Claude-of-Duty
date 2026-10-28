import * as THREE from 'three';

import { hdrTarget, blit } from './pass.js';
import { CascadedShadowMaps } from './csm.js';
import { MaterialPatcher } from './materialpatch.js';
import { GBuffer } from './prepass.js';
import { Gtao } from './gtao.js';
import { ContactShadows } from './contact.js';
import { Ssr } from './ssr.js';
import { Taa } from './taa.js';
import { MotionBlur } from './motionblur.js';
import { DepthOfField } from './dof.js';
import { Bloom } from './bloom.js';
import { AutoExposure } from './exposure.js';
import { createGradeLut } from './lut.js';
import { createComposite, createFxaa, createDebug, createViewComposite } from './composite.js';
import { buildFallbackEnvironment } from './env.js';
import { RenderProbeScene } from './probe.js';

const QUALITY_LEVEL = { low: 0, medium: 1, high: 2, ultra: 3 };

/**
 * Registration range at or below which a punctual light counts as a room/street
 * PRACTICAL rather than as an effect flash. See `settings.practicalGain`.
 */
const PRACTICAL_RANGE = 30;

// Full-daylight key intensity (SUN_ILLUMINANCE_TOP through a clear atmosphere).
// Only used to normalise the viewmodel rig, never to light anything.
const REF_DAYLIGHT = 4.6;

/**
 * OVERWATCH renderer.
 *
 * Frame order (everything HDR, linear, float, until the very last write):
 *
 *   1  scene walk        collect draw/hide lists, patch new materials
 *   2  CSM               N stabilised cascades into one R32F array texture
 *   3  jitter            sub-pixel offset on the WORLD camera for TAA
 *   4  prepass           MRT: view normal + velocity + linear depth
 *   5  GTAO              horizon-arc AO, temporally accumulated
 *   6  contact shadows   short depth-buffer ray march toward the sun
 *   7  SSR               marched against depth, coloured from last frame
 *   8  forward world     lit with 2/5/6/7 injected into every material
 *   9  viewmodel         same lighting, into its OWN MSAA colour+depth target
 *  10  TAA               velocity reprojection + YCoCg variance clipping
 *  11  motion blur       velocity-tile reconstruction filter
 *  12  ADS depth of field gather CoC blur, only while the sights are up
 *  13  custom passes     whatever fx/ui/sky registered
 *  14  viewmodel resolve premultiplied composite over the world, FXAA'd
 *  15  metering          GPU log-luminance reduction -> EV100 -> exposure
 *  16  bloom             Karis pyramid with a soft-knee highlight threshold
 *  17  composite         AgX + LUT + vignette + CA + grain -> sRGB
 *  18  FXAA              only when TAA is off
 *
 * Why the viewmodel is resolved separately (steps 9 + 14): everything in
 * `viewScene` moves in VIEW space — the ADS transition, sway, bob, recoil — and
 * a velocity buffer built from camera view-projection matrices describes none of
 * it. Those pixels emitted zero motion, so TAA reprojected them onto a stale
 * history sample holding the static background and blended it in at ~85%. That
 * is what made the optic tube, the mount pedestal and the glove semi-transparent
 * with balcony rails and power lines legible straight through them. Compositing
 * after the resolve removes the whole failure mode instead of tuning around it,
 * and as a bonus keeps the weapon out of the volumetric fog and the ADS DOF.
 *
 * Nothing in the chain uses three's examples/jsm post stack.
 *
 * ---------------------------------------------------------------------------
 * PUBLIC API — `const r = ctx.get('render')`
 * ---------------------------------------------------------------------------
 *   r.renderer                THREE.WebGLRenderer (do not change state mid-frame)
 *   r.screenSize              { width, height } of the internal HDR target
 *   r.displaySize             { width, height } of the canvas backbuffer
 *   r.depthTexture            R32F linear view depth in METRES (positive)
 *   r.velocityTexture         RG16F screen-space velocity as a UV delta
 *   r.normalTexture           RGBA16F oct-encoded VIEW normal (xy), coverage (z:
 *                             1 = static geometry, 0.7 = skinned/morphed, 0 =
 *                             nothing; test against 0.5 for "is there a
 *                             surface"), material id (w)
 *   r.aoTexture               R16F GTAO visibility, or null
 *   r.exposureTexture         1x1 float, .r = exposure scalar, .g = EV100
 *   r.hdrTexture              the pre-post HDR colour target
 *   r.prewarmMaterials()      compile the shadow/prepass/post programs up front
 *                             (loading screen only; advances no simulation)
 *   r.registerPass(pass)      pass.render(renderer, inTex, outTarget, r);
 *                             optional .order, .enabled, .resize(w,h).
 *                             Returns an unregister function.
 *   r.addLight(light, opts)   register a punctual light for distance culling
 *   r.requestEnvMap()         the PMREM environment currently in use
 *   r.setEnvMap(tex)          install a new PMREM (sky subsystem)
 *   r.patchMaterials(root)    force-inject shadows/AO/SSR into new materials
 *                             (happens automatically within a frame anyway)
 *   r.setExposureBias(ev)     +1 EV = one stop darker
 *   r.settings                live tuning: bloomStrength, bloomThreshold,
 *                             bloomKnee, vignette, adsVignette, grain,
 *                             chromatic, sharpen, shutter, aoRadius,
 *                             aoIntensity, contactLength, contactStrength,
 *                             shadowStrength, sunSoftness,
 *                             exposureKey, autoExposure, skyFill, groundFill,
 *                             bounceFill, viewKeyScale/Max/Gamma,
 *                             viewFillRatio, viewRimRatio, viewHemiRatio,
 *                             viewFillOcclusion, dofMaxCoc, dofNearRatio,
 *                             dofFocusMin/Max, dofFarStart, dofFarRange,
 *                             dofNearScale
 *
 * Note on colour space: chromatic aberration, bloom and the cos^4 lens vignette
 * are LINEAR-light lens effects and happen before the tone map — a vignette
 * applied to code values is a flat multiply that makes display white
 * unreachable everywhere but the centre of the frame. The composite then
 * tone-maps, encodes to sRGB, and applies the grade LUT, grain and dither in
 * DISPLAY space: the LUT is authored display-referred (additive toe and
 * split-tone offsets in code values), so anything added to that tail belongs
 * after the encode too.
 *   r.debugView               'ao'|'normal'|'velocity'|'depth'|'ssr'|'ssrmask'|
 *                             'contact'|'bloom'|'view'|'viewalpha'|null
 *                             (also ?rview=)
 *
 * Per-object opt-outs, set on the Object3D:
 *   userData.owNoPrepass = true   keep out of depth/normal/velocity (particles)
 *   userData.owNoShadow  = true   do not cast into the cascades
 *   userData.owMatId     = 0..1   written to the gbuffer alpha for custom fx
 * Transparent materials are excluded from the prepass and shadows automatically.
 */
export class RenderSystem {
  static id = 'render';
  static deps = [];

  async init(ctx) {
    this.ctx = ctx;
    const cfg = ctx.config;
    const q = cfg.q;
    this.q = q;
    this.qLevel = QUALITY_LEVEL[cfg.quality] ?? 3;
    this.rng = ctx.rng.fork();
    this.frame = 0;

    // Low preset only: dynamically trade internal resolution for frame pacing.
    // Use the engine's frame delta (not GPU timer queries, which are not
    // consistently available in browsers). Ignore tab-switch/debugger hitches.
    this._baseRenderScale = q.renderScale;
    this._dynamicResolution = this.qLevel === 0 && !cfg.deterministic;
    this._dynamicFrameMs = 16.7;
    this._dynamicScaleFrames = 0;

    // ---- renderer -------------------------------------------------------
    const renderer = new THREE.WebGLRenderer({
      canvas: ctx.canvas,
      antialias: false, // TAA/FXAA handle this; MSAA cannot resolve HDR post
      alpha: false,
      depth: true,
      stencil: false,
      premultipliedAlpha: false,
      preserveDrawingBuffer: false,
      powerPreference: 'high-performance',
      failIfMajorPerformanceCaveat: false,
    });
    if (!renderer.capabilities.isWebGL2) {
      throw new Error('[render] WebGL2 is required');
    }
    renderer.autoClear = false;
    renderer.autoClearColor = false;
    renderer.autoClearDepth = false;
    renderer.info.autoReset = false;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.NoToneMapping; // we tonemap in the composite
    renderer.shadowMap.enabled = true; // for spot/point lights owned by others
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.shadowMap.autoUpdate = true;
    renderer.setClearColor(0x000000, 1);
    this.renderer = renderer;

    // ---- make every pre-compile see the FINAL program -----------------------
    // MEASURED: 26 of 144 live programs were unpatched duplicates of a lit
    // material — compiled once by a caller that pre-compiles the world scene
    // (src/core/prewarm.js does exactly this), then thrown away and compiled
    // again the moment the first real frame walked the scene and injected the
    // shadow/AO/SSR chunk, because `patch()` sets `needsUpdate`. That is 18% of
    // the boot's compile budget spent on programs that are never used to draw
    // anything. Patching before the compile makes the pre-compiled program the
    // one the frame loop actually wants. Restricted to the world and viewmodel
    // scenes so a subsystem baking a texture through its own scene is untouched.
    const rawCompile = renderer.compile.bind(renderer);
    renderer.compile = (target, cam, targetScene) => {
      if (target === ctx.scene) this._patchLikeFrame(ctx.scene, false);
      else if (target === ctx.viewScene) this._patchLikeFrame(ctx.viewScene, true);
      return rawCompile(target, cam, targetScene);
    };

    this.maxAnisotropy = Math.min(
      q.anisotropy,
      renderer.capabilities.getMaxAnisotropy()
    );

    // ---- subsystems of the pipeline --------------------------------------
    this.csm = new CascadedShadowMaps(renderer, {
      cascades: q.cascades,
      mapSize: q.shadowMapSize,
      maxDistance: q.shadowDistance,
    });
    this.patcher = new MaterialPatcher(this.csm.uniforms, {
      cascades: this.csm.cascades,
      quality: this.qLevel,
    });

    this.gbuffer = new GBuffer();
    this.gtao = q.gtao ? new Gtao() : null;
    this.contact = this.qLevel >= 1 ? new ContactShadows() : null;
    this.ssr = q.ssr ? new Ssr() : null;
    this.taa = q.taa ? new Taa() : null;
    this.motionBlur = q.motionBlur ? new MotionBlur() : null;
    // ADS depth of field. Cheap (half-res gather, 32 taps) and only ever runs
    // while the sights are actually up, so it costs nothing in hipfire.
    this.dof = this.qLevel >= 1 ? new DepthOfField() : null;
    this.bloom = q.bloom ? new Bloom(this.qLevel >= 2 ? 6 : 5) : null;
    this.exposure = new AutoExposure();
    // Headroom for a physically-scaled sky (sunlit scenes reach ~5000 cd/m2).
    // The lower limit is the night exposure lock: a moonlit street meters at
    // EV100 -5.2, and letting the meter chase that turns night into an overcast
    // afternoon. Daylight shots meter between -1 and -2.1, so this only ever
    // binds after dark.
    this.exposure.setLimits(-4.3, 20);
    this.lut = createGradeLut('default');
    this.composite = createComposite(this.lut);
    this.viewComposite = createViewComposite();
    this.fxaa = q.taa ? null : createFxaa();
    // MSAA on the viewmodel target only. It is the one buffer whose geometric
    // edges no longer get a temporal filter, and 4x on a single small pass is
    // far cheaper than any spatial substitute at the same quality.
    this._viewSamples = this.qLevel >= 2 ? 4 : this.qLevel >= 1 ? 2 : 0;

    // Always on: depthTexture/velocityTexture are part of the public contract
    // (soft particles, SSR, motion blur) even when our own effects are off.
    this.needsPrepass = true;

    this.hdrRt = null;
    this.viewRt = null;
    this.ldrRt = null;
    this.pingRt = [null, null];
    this._pingIndex = 0;
    this._adsT = 0;
    this._weapons = null;

    this._tmpV3 = new THREE.Vector3();
    this._tmpV3b = new THREE.Vector3();
    this._fillHue = new THREE.Vector3();
    this._fillHue2 = new THREE.Vector3();
    this._fillSkySave = new THREE.Vector3();
    this._fillGroundSave = new THREE.Vector3();
    this._ambLevel = 0.6;
    this._roomsReady = false;
    this._skyExposureBias = 0;

    // ---- lighting defaults ------------------------------------------------
    // The sky subsystem normally owns the sun. Until it does, we provide one
    // so the world is never lit by nothing; it hides itself the moment a
    // brighter directional light shows up in the scene.
    this.sun = new THREE.DirectionalLight(0xffe8c4, 4.3);
    this.sun.name = 'ow-fallback-sun';
    this.sun.position.set(-42, 46, 26);
    this.sun.castShadow = false;
    this.sun.target.position.set(0, 0, 0);
    ctx.scene.add(this.sun);
    ctx.scene.add(this.sun.target);
    this.activeSun = this.sun;
    this.sunDir = new THREE.Vector3(0, 1, 0);
    this.sunDirView = new THREE.Vector3(0, 1, 0);

    // ---- viewmodel light rig ---------------------------------------------
    // The weapon lives in its own scene. Handing it one copy of the world sun
    // means that whenever the sun is behind the gun the camera-facing side gets
    // nothing, and weapon albedos are physically correct (anodised aluminium is
    // 0.026 linear) so it goes to a black silhouette. It gets a real 3-point
    // rig instead, with every direction fixed in VIEW space so the weapon reads
    // identically at any world sun azimuth — which is what every shipped FPS
    // does, and the reason their guns are always legible.
    this.viewSun = new THREE.DirectionalLight(0xffe8c4, 2.0);
    this.viewSun.name = 'ow-viewmodel-key';
    this.viewKeyFill = new THREE.DirectionalLight(0x9ec4ff, 0.6);
    this.viewKeyFill.name = 'ow-viewmodel-fill';
    this.viewRim = new THREE.DirectionalLight(0xffd7a8, 1.0);
    this.viewRim.name = 'ow-viewmodel-rim';
    this.viewFill = new THREE.HemisphereLight(0x8fb6ff, 0x36302a, 0.35);
    // Warm bounce off the ground/street, arriving from BELOW. Without it, any
    // part of the weapon or hands that sits in the gun's own cast shadow — the
    // support glove under the handguard is the worst case — is lit by nothing
    // but the cool sky fill, so a warm glove albedo still renders blue. A real
    // street throws a stop of warm light back up; this is that term.
    this.viewBounce = new THREE.DirectionalLight(0xffb87a, 0.5);
    this.viewBounce.name = 'ow-viewmodel-bounce';
    // View-space directions the light arrives FROM: key upper-front-left,
    // fill lower-front-right, rim from behind to catch the top edges of the
    // receiver, rail and optic body, bounce from below-front.
    this._viewKeyDir = new THREE.Vector3(-0.45, 0.75, 0.55).normalize();
    this._viewFillDir = new THREE.Vector3(0.6, -0.15, 0.5).normalize();
    this._viewRimDir = new THREE.Vector3(0.2, 0.35, -0.9).normalize();
    this._viewBounceDir = new THREE.Vector3(-0.2, -0.86, 0.47).normalize();
    this._tmpV3c = new THREE.Vector3();
    for (const l of [this.viewSun, this.viewKeyFill, this.viewRim, this.viewBounce]) {
      l.castShadow = false;
      ctx.viewScene.add(l, l.target);
    }
    ctx.viewScene.add(this.viewFill);
    // The frame loop skips the viewmodel pass when nothing but our own rig is
    // in there; remember how many children that is.
    this._viewRigChildren = ctx.viewScene.children.length;

    const env = buildFallbackEnvironment(renderer, this._dirFromLight(this.sun, this.sunDir));
    this.envTarget = env.target;
    this.envEquirect = env.equirect;
    this.envMap = this.envTarget.texture;
    if (!ctx.scene.environment) ctx.scene.environment = this.envMap;
    if (!ctx.scene.background) ctx.scene.background = this.envEquirect;
    ctx.viewScene.environment = ctx.scene.environment;
    this._assignedViewEnv = ctx.scene.environment;

    // ---- bookkeeping ------------------------------------------------------
    this.passes = [];
    this.lights = [];
    this._draw = [];
    this._nDraw = 0;
    this._hide = [];
    this._nHide = 0;
    this._noShadow = [];
    this._nNoShadow = 0;
    this._dirLights = [];
    this._nDirLights = 0;
    this._foreignMeshes = 0;

    this._currVP = new THREE.Matrix4();
    this._prevVP = new THREE.Matrix4();
    this._invVP = new THREE.Matrix4();
    this._camPos = new THREE.Vector3();
    this._jitterSaved = new THREE.Vector2();
    this._viewVisible = false;
    this._readback = new Float32Array(4);
    this._readback2 = new Float32Array(4);
    this._jittered = false;
    this._firstFrame = true;

    this.screenSize = { width: 1, height: 1 };
    this.displaySize = { width: 1, height: 1 };
    this.depthTexture = null;
    this.velocityTexture = null;
    this.normalTexture = null;
    this.aoTexture = null;
    this.exposureTexture = this.exposure.texture;

    this.settings = {
      exposureBias: 0, // EV; positive = darker
      exposureKey: 1.06,
      autoExposure: true,
      // The pyramid is ADDED now, not mixed, and it is soft-knee thresholded at
      // `bloomThreshold` in exposure-scaled linear light — so this is the gain on
      // light that is genuinely above display white (sun disc, glints, muzzle
      // flash), not a whole-frame veiling-glare percentage. 0.048 of a mix() was
      // invisible; 0.12 of an additive thresholded pyramid is a specular event.
      bloomStrength: 0.14,
      // 1.6, not 0.85. A daylight sky lands around 1.0-1.5 in exposure-scaled
      // linear light, so at 0.85 the SKY was the brightest thing in the pyramid
      // and the widest mip smeared it four or five pixels over every roofline and
      // every silhouette in front of it — an enemy on a balcony measured 3%
      // contrast against the cloud he was standing in front of. At 1.6 the sky is
      // below the knee and the pyramid is back to what it is for: the sun disc,
      // glints, tracers, muzzle flash.
      bloomThreshold: 1.6,
      bloomKnee: 0.9,
      // Lateral CA. Kept small: it is a lens signature, and at 0.0018 the R/B
      // split reached most of a pixel in the corners, which the sharpen filter
      // then turned into visible fringing on every high-contrast edge.
      chromatic: 0.0011,
      // Vignette and grain are applied in DISPLAY space (see composite.js), so
      // these are code-value amplitudes, not linear-light ones.
      vignette: 0.24,
      // Closes in while the sights are up: the frame has to tell you your eye is
      // behind a tube, not just that the gun moved.
      adsVignette: 0.34,
      grain: 0.010,
      // ---- ADS depth of field (see dof.js) ---------------------------------
      // maxCoc is in pixels at 1080p and is reached well beyond focusMax, so
      // geometry past ~25 m goes visibly soft while the optic — composited after
      // the pass — stays pin sharp.
      // 3.3 px at 1080p, down 40%: at 5.5 the near and mid ground of an ADS frame
      // was a watercolour smear that hid the very thing the sights are pointed at.
      dofMaxCoc: 3.3,
      dofNearRatio: 0.38,
      dofFocusMin: 3.0,
      dofFocusMax: 18.0,
      dofFarStart: 1.15,
      dofFarRange: 18.0,
      dofNearScale: 0.55,
      sharpen: 0.25,
      lutStrength: 1.0,
      shutter: 0.42,
      aoRadius: 1.35,
      // Occlusion is a shaping tool, not a darkening tool: 1.7 with no bounce
      // fill behind it was what turned every corner into a black hole.
      aoIntensity: 1.1,
      // Screen-space contact shadows. The length is a world-space ray in metres
      // at 1x distance scaling (see contact.js): it has to resolve the 0-40 cm
      // range, because that is the gap a cascade texel cannot see and the
      // difference between a crate sitting on the floor and a crate stickered
      // onto it. Debuggable with ?rview=contact.
      contactLength: 0.4,
      contactStrength: 1.0,
      // Hemispheric sky/ground fill, and the warm anti-sun bounce wrap, as a
      // fraction of the key's intensity (see _updateBounceFill).
      //
      // These are a *budget*, and the budget is what sets the key:fill ratio.
      // Real direct sun runs 5-8:1 against its own shade; at skyFill 0.09 /
      // groundFill 0.055 / iblDiffuse 1.0 the indirect terms were supplying 42%
      // of every lit value and the ratio collapsed to 2.4:1 — a frame with no
      // sun in it, only a bright ambient. Cut hard and let the beam carry it.
      // Cool skylight band. This is the frame's ONLY strongly chromatic
      // indirect term — the PMREM integrates a blue upper hemisphere against a
      // warm ground plane and comes back very nearly achromatic on a vertical
      // wall — so it is deliberately the biggest one now, and iblDiffuse below
      // came down by the same amount to pay for it. Measured on the 16:30
      // frame: a shaded facade went from B-R = 0.0002 (4.5% saturation, "dead
      // neutral regardless of surroundings") to a legible cool cast.
      skyFill: 0.32,
      // Warm bounce off the street, onto soffits, undersides and low faces.
      groundFill: 0.013,
      // ...and the wrap term: the shaded side of the street lit by the sunlit
      // side of it. Both up hard, because this is the "warm kick where a shadow
      // faces a sunlit surface" that was missing entirely.
      bounceFill: 0.008,
      // The PMREM sky cubemap is the single biggest indirect term in the frame
      // (materials ship envMapIntensity 1.6). Scaling its *diffuse* here is the
      // only place the total indirect budget can actually be controlled from.
      // Specular radiance is left alone — that is reflection, not fill.
      iblDiffuse: 0.030,
      // Indirect floor inside a coarse interior volume. Skylight does not reach
      // the middle of a closed room; without this the doorway reads as a hole
      // cut in a card because the room is brighter than the street outside it.
      interiorIndirect: 0.035,
      // Global trim on room and street practicals (see PRACTICAL_RANGE).
      //
      // Twenty interior bulbs and twenty-two sodium lamps are the ONLY light in
      // a closed room and by far the loudest thing in a night street, so they
      // are what actually sets two of this renderer's headline ratios:
      // interior-to-exterior through a doorway, and warm-pool-to-cool-ambient
      // after dark. At unity an interior metered within 1.3 stops of the sunlit
      // facade framed in its own opening (a real one runs 4-5) and every
      // surface in the night frame took its hue from a lamp. Half a stop off
      // them buys most of both, and it is applied here rather than at the
      // source because the balance is a lighting decision, not an art one.
      practicalGain: 0.55,
      // Sky the viewmodel can actually see, past the shooter's own body.
      viewFillOcclusion: 0.45,
      // Viewmodel 3-point rig. The key is scaled off the scene's own light
      // level (see _updateViewRig); fill, rim and hemisphere are ratios of it.
      viewKeyScale: 0.55,
      viewKeyMax: 2.6,
      viewFillRatio: 0.3,
      viewRimRatio: 0.5,
      // 0.35 hemisphere against a ~2.2 daylight key, expressed as a ratio so it
      // follows the time of day instead of blowing the gun out at night.
      viewHemiRatio: 0.16,
      // Warm ground bounce from below. Sized to lift the glove out of the
      // handguard's cast shadow without competing with the key: at 0.34 of the
      // key it is ~1.5 stops down, which is about what a sand street returns.
      viewBounceRatio: 0.34,
      viewKeyGamma: 0.65,
      shadowStrength: 1.0,
      sunSoftness: 0.024,
    };
    this._applySettings();

    this.probe = new RenderProbeScene(this.rng.fork());
    this.probeActive = false;
    // In capture mode print the metering chain so a bad exposure is obvious
    // in the harness log rather than something to guess at from the PNG.
    this._probeExposure = ctx.config.deterministic === true;
    this.debugView = new URLSearchParams(location.search).get('rview') || null;
    this._debugPass = null;

    // `?owNoCascadeCull=1` puts every caster back into every cascade. Kept as
    // the A/B switch the pixel gate was run through, and the escape hatch if a
    // subsystem ever ships geometry whose bounds lie about where it is.
    this._noCascadeCull = /[?&]owNoCascadeCull=1/.test(location.search);
    this._visit = this._visit.bind(this);
    this._visitView = this._visitView.bind(this);

    const w = ctx.canvas.clientWidth || 1920;
    const h = ctx.canvas.clientHeight || 1080;
    this.resize(w, h, ctx);

    console.info(
      `[render] WebGL2 · ${cfg.quality} · ${this.csm.cascades}x${this.csm.mapSize} CSM · ` +
        `taa:${!!this.taa} gtao:${!!this.gtao} ssr:${!!this.ssr} mb:${!!this.motionBlur}`
    );
  }

  // ==========================================================================
  //  public API (see ARCHITECTURE.md "Render integration")
  // ==========================================================================

  /**
   * Insert a custom post pass.
   * `pass.render(renderer, inputTexture, outputTarget, renderSystem)` must
   * write a full-screen result into `outputTarget`.
   * `pass.order` (default 0) controls ordering; `pass.enabled !== false`.
   */
  registerPass(pass) {
    this.passes.push(pass);
    this.passes.sort((a, b) => (a.order ?? 0) - (b.order ?? 0));
    if (pass.resize) pass.resize(this.screenSize.width, this.screenSize.height);
    return () => {
      const i = this.passes.indexOf(pass);
      if (i >= 0) this.passes.splice(i, 1);
    };
  }

  /** Register a punctual light so it participates in culling and budgets. */
  addLight(light, opts = {}) {
    if (!light || this.lights.some((l) => l.light === light)) return light;
    this.lights.push({
      light,
      range: opts.range ?? light.distance ?? 25,
      priority: opts.priority ?? 1,
      baseIntensity: light.intensity,
    });
    return light;
  }

  removeLight(light) {
    const i = this.lights.findIndex((l) => l.light === light);
    if (i >= 0) this.lights.splice(i, 1);
  }

  /** The PMREM environment currently in use. */
  requestEnvMap() {
    return this.ctx?.scene.environment ?? this.envMap;
  }

  /** Let the sky subsystem hand us its PMREM. */
  setEnvMap(texture) {
    this.ctx.scene.environment = texture;
    this.ctx.viewScene.environment = texture;
    this.envMap = texture;
  }

  /** Force every lit material in a subtree to be patched immediately. */
  patchMaterials(root) {
    root.traverse((o) => {
      const m = o.material;
      if (!m) return;
      if (Array.isArray(m)) for (const mm of m) this.patcher.patch(mm);
      else this.patcher.patch(m);
    });
  }

  /**
   * Patch exactly the set of materials the frame loop's own scene walk would
   * patch, and no more.
   *
   * `patchMaterials()` is the public, deliberately-broad version: it reaches
   * every material in a subtree. That is the wrong tool for pre-compiling,
   * because injecting the shadow/AO/fill chunk into a material the frame loop
   * never patches CHANGES HOW THAT MATERIAL SHADES — measured at 0.04% of
   * pixels, up to 26/255, when this used `traverse()`. Mirroring `_visit` /
   * `_visitView` exactly (same traversal, same object-type predicate) makes
   * pre-patching a pure reordering of *when* the identical set is patched.
   */
  _patchLikeFrame(root, isViewScene) {
    root.traverseVisible((o) => {
      if (isViewScene) {
        if (o.isMesh !== true) return;
      } else if (
        o.isMesh !== true &&
        o.isPoints !== true &&
        o.isSprite !== true &&
        o.isLine !== true
      ) {
        return;
      }
      const m = o.material;
      if (Array.isArray(m)) for (let i = 0; i < m.length; i++) this.patcher.patch(m[i]);
      else if (m) this.patcher.patch(m);
    });
  }

  setExposureBias(ev) {
    this.settings.exposureBias = ev;
  }

  /**
   * Compile every program this subsystem can reach, without drawing a gameplay
   * frame. Call it from the loading screen (src/core/prewarm.js).
   *
   * WHY THE ENGINE'S OWN PRE-WARM IS NOT ENOUGH — measured, not guessed.
   * `renderer.compileAsync(scene, camera)` only reaches the *forward lit*
   * program of each material. It does not reach:
   *
   *   - the CSM depth variant of a material (skinned / morphed / instanced /
   *     batched are four separate programs off one ShaderMaterial),
   *   - the MRT prepass variant, same four,
   *   - a single one of the ~25 full-screen post programs, because those are
   *     not in any scene graph.
   *
   * Those are exactly the ones that used to land mid-play: profiling showed up
   * to 30 programs compiling on ONE frame, and that frame took 3.1-3.9 s on a
   * cold shader cache.
   *
   * NOTHING HERE ADVANCES THE SIMULATION. It never calls `engine.step()`, never
   * touches `time`, `rng`, the TAA history, the exposure adaptation buffers,
   * the velocity history or `this.frame`. It draws into the shadow array, the
   * gbuffer and a 4x4 scratch target — all three of which the next real frame
   * overwrites in full before anything reads them — so it is invisible to the
   * pixel-diff gate.
   *
   * PIXEL GATE, measured with tools/baseline.mjs at 1920x1080 on all 11 shots:
   * the patch + `compileAsync` step and the post-chain step are bit-identical.
   * The depth/shadow step is NOT, when it is called after frames have already
   * been drawn: on the `night` shot it moves 26 pixels by 2/255, which survives
   * snapshotting and restoring the whole cascade fit and the sun takeover, so
   * the residue is in the shadow-array / gbuffer contents themselves. It is
   * therefore only ON by default at `frame === 0` — before a single frame has
   * been drawn there is no cascade fit, no gbuffer and no shadow array to
   * disturb — and has to be asked for explicitly at any other time.
   *
   * @param {object}  [opts]
   * @param {boolean} [opts.post=true]   compile the full-screen pass chain
   * @param {boolean} [opts.shadow]      compile the CSM depth + prepass variants;
   *                                     defaults to true only before frame 1
   * @returns {Promise<object>} { ok, ms, programsBefore, programsAfter, compiled }
   */
  async prewarmMaterials({ post = true, shadow = this.frame === 0 } = {}) {
    const t0 = performance.now();
    const renderer = this.renderer;
    const ctx = this.ctx;
    if (!renderer || !ctx) return { ok: false, reason: 'not initialised' };
    const programsBefore = renderer.info.programs?.length ?? 0;
    const prevTarget = renderer.getRenderTarget();

    try {
      // 1. Patch first, ALWAYS. A program compiled off an unpatched material is
      //    thrown away by the first frame that walks the scene (see the
      //    renderer.compile wrapper in init) — pure waste of the boot budget.
      this._patchLikeFrame(ctx.scene, false);
      this._patchLikeFrame(ctx.viewScene, true);

      // 2. Forward lit programs. compileAsync uses KHR_parallel_shader_compile
      //    where the driver has it, so this does not block the main thread.
      try {
        await renderer.compileAsync(ctx.scene, ctx.camera);
        await renderer.compileAsync(ctx.viewScene, ctx.viewCamera);
      } catch {
        renderer.compile(ctx.scene, ctx.camera);
        renderer.compile(ctx.viewScene, ctx.viewCamera);
      }

      // 3. Depth-only variants. There is no compile-time API for an override
      //    material, so the only way to reach them is to actually run the two
      //    depth passes once — which is cheap, writes only to buffers the next
      //    frame clears, and advances nothing.
      if (shadow) {
        const camera = ctx.camera;
        camera.updateMatrixWorld();
        this._collect(ctx.scene);
        this._syncSun(camera);
        const bg = ctx.scene.background;
        // Fitting the cascades out of frame is not free: `update()` leaves a fit
        // behind and the next frame's refit does not fully overwrite it (see
        // CascadedShadowMaps.snapshotFit — measured at 1.3 M pixels, up to
        // 26/255). Take the fit, use it, hand it straight back.
        const fit = this.csm.snapshotFit();
        // ...and the same for the sun takeover `_syncSun` performs, so a prewarm
        // cannot hand the next frame a different active sun / fallback state
        // than the one it would have computed for itself.
        const sunSave = {
          dir: this.sunDir.clone(),
          dirView: this.sunDirView.clone(),
          active: this.activeSun,
          fallbackVisible: this.sun.visible,
          ambLevel: this._ambLevel,
        };
        ctx.scene.background = null;
        this._hideList(this._hide, this._nHide);
        this._hideList(this._noShadow, this._nNoShadow);
        this.csm.update(camera, this.sunDir, this.settings.sunSoftness);
        this.csm.render(renderer, ctx.scene, this._draw, this._nDraw);
        this._showList(this._noShadow, this._nNoShadow);
        this.gbuffer.render(
          renderer,
          ctx.scene,
          camera,
          this._currVP.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse),
          this._currVP,
          true
        );
        this._showList(this._hide, this._nHide);
        ctx.scene.background = bg;
        this.csm.restoreFit(fit);
        this.sunDir.copy(sunSave.dir);
        this.sunDirView.copy(sunSave.dirView);
        this.activeSun = sunSave.active;
        this.sun.visible = sunSave.fallbackVisible;
        this._ambLevel = sunSave.ambLevel;
      }

      // 4. The post chain. A pass's program does not depend on the size of what
      //    it is drawn into, so a 4x4 scratch target compiles it for free.
      if (post) {
        const scratch = hdrTarget(4, 4, { name: 'prewarm-scratch' });
        const mats = [];
        this._collectPassMaterials(mats);
        for (const m of mats) {
          try {
            blit(renderer, m, scratch);
          } catch {
            /* a pass with an unsatisfiable uniform must not stop the rest */
          }
        }
        scratch.dispose();
      }
    } catch (e) {
      return { ok: false, reason: String(e && e.message ? e.message : e) };
    } finally {
      renderer.setRenderTarget(prevTarget);
    }

    const programsAfter = renderer.info.programs?.length ?? 0;
    return {
      ok: true,
      ms: Math.round(performance.now() - t0),
      programsBefore,
      programsAfter,
      compiled: programsAfter - programsBefore,
      parallel: !!renderer.getContext().getExtension('KHR_parallel_shader_compile'),
    };
  }

  /** Every full-screen material this subsystem owns, for prewarmMaterials(). */
  _collectPassMaterials(out) {
    const add = (p) => {
      if (p && p.material) out.push(p.material);
    };
    add(this.composite);
    add(this.viewComposite);
    add(this.fxaa);
    if (this.gtao) {
      add(this.gtao.core);
      add(this.gtao.temporal);
      add(this.gtao.blur);
    }
    if (this.contact) {
      add(this.contact.pass);
      add(this.contact.blur);
    }
    if (this.ssr) {
      add(this.ssr.pass);
      add(this.ssr.blur);
    }
    if (this.taa) add(this.taa.pass);
    if (this.motionBlur) {
      add(this.motionBlur.tilePass);
      add(this.motionBlur.blurPass);
    }
    if (this.dof) {
      add(this.dof.pre);
      add(this.dof.gather);
      add(this.dof.combine);
    }
    if (this.bloom) {
      add(this.bloom.down);
      add(this.bloom.up);
    }
    add(this.exposure.logPass);
    add(this.exposure.reducePass);
    add(this.exposure.adaptPass);
    return out;
  }

  get hdrTexture() {
    return this.hdrRt?.texture ?? null;
  }

  /**
   * Read back a block of the pre-post HDR buffer, in SCENE RADIANCE UNITS
   * (i.e. before exposure). Diagnostic only — this stalls the pipeline, so it
   * is never called from a frame. Coordinates are fractions of the screen.
   *
   * This exists because every tone/lighting argument in this subsystem is an
   * argument about *ratios of scene radiance* — key:fill, sky:sunlit-wall,
   * interior:exterior — and reading them off a graded PNG means inverting the
   * tone curve by hand and getting it wrong.
   */
  probeHdr(u0, v0, u1, v1) {
    const rt = this.hdrRt;
    if (!rt) return null;
    const W = this.screenSize.width;
    const H = this.screenSize.height;
    const x = Math.max(0, Math.round(u0 * W));
    // GL origin is bottom-left; screen v is top-down.
    const y = Math.max(0, Math.round((1 - v1) * H));
    const w = Math.max(1, Math.min(W - x, Math.round((u1 - u0) * W)));
    const h = Math.max(1, Math.min(H - y, Math.round((v1 - v0) * H)));
    const half = rt.texture.type === THREE.HalfFloatType;
    const buf = half ? new Uint16Array(w * h * 4) : new Float32Array(w * h * 4);
    this.renderer.readRenderTargetPixels(rt, x, y, w, h, buf);
    const dec = half ? THREE.DataUtils.fromHalfFloat : (v) => v;
    let r = 0;
    let g = 0;
    let b = 0;
    let mx = 0;
    const n = w * h;
    for (let i = 0; i < n; i++) {
      const cr = dec(buf[i * 4]);
      const cg = dec(buf[i * 4 + 1]);
      const cb = dec(buf[i * 4 + 2]);
      r += cr;
      g += cg;
      b += cb;
      mx = Math.max(mx, cr, cg, cb);
    }
    return { r: r / n, g: g / n, b: b / n, max: mx, n };
  }

  /** Whole HDR buffer box-downsampled to cols x rows, row 0 = top. Diagnostic. */
  probeHdrGrid(cols = 32, rows = 18) {
    const rt = this.hdrRt;
    if (!rt) return null;
    const W = this.screenSize.width;
    const H = this.screenSize.height;
    const half = rt.texture.type === THREE.HalfFloatType;
    const buf = half ? new Uint16Array(W * H * 4) : new Float32Array(W * H * 4);
    this.renderer.readRenderTargetPixels(rt, 0, 0, W, H, buf);
    const dec = half ? THREE.DataUtils.fromHalfFloat : (v) => v;
    const out = new Float32Array(cols * rows * 3);
    const cnt = new Float32Array(cols * rows);
    for (let y = 0; y < H; y++) {
      const gy = Math.min(rows - 1, Math.floor(((H - 1 - y) / H) * rows));
      for (let x = 0; x < W; x++) {
        const gx = Math.min(cols - 1, Math.floor((x / W) * cols));
        const s = (y * W + x) * 4;
        const d = (gy * cols + gx) * 3;
        out[d] += dec(buf[s]);
        out[d + 1] += dec(buf[s + 1]);
        out[d + 2] += dec(buf[s + 2]);
        cnt[gy * cols + gx]++;
      }
    }
    const res = [];
    for (let i = 0; i < cols * rows; i++) {
      const c = Math.max(1, cnt[i]);
      res.push([out[i * 3] / c, out[i * 3 + 1] / c, out[i * 3 + 2] / c]);
    }
    return { cols, rows, cells: res };
  }

  _applySettings() {
    const s = this.settings;
    const cu = this.composite.uniforms;
    cu.uLens.value.set(s.chromatic, s.vignette, s.grain, 0);
    cu.uGrade.value.set(s.bloomStrength, s.lutStrength, this.taa ? s.sharpen : 0, this.lut.size);
    this.csm.setStrength(s.shadowStrength);
    if (this.bloom) {
      this.bloom.threshold = s.bloomThreshold;
      this.bloom.knee = s.bloomKnee;
    }
    if (this.gtao) {
      this.gtao.setRadius(s.aoRadius);
      this.gtao.setIntensity(s.aoIntensity);
    }
    if (this.contact) {
      this.contact.setLength(s.contactLength);
      this.contact.setStrength(s.contactStrength);
    }
  }

  // ==========================================================================
  //  sizing
  // ==========================================================================

  resize(w, h, ctx) {
    // UHD 770 is bandwidth-limited: avoid rendering a high-DPI backbuffer
    // in the integrated-GPU preset. Higher quality presets retain capped HiDPI.
    const pr = Math.min(globalThis.devicePixelRatio || 1, this.qLevel === 0 ? 1 : 1.5);
    this.renderer.setPixelRatio(pr);
    this.renderer.setSize(w, h, false);

    const dw = Math.max(1, Math.floor(w * pr));
    const dh = Math.max(1, Math.floor(h * pr));
    const rw = Math.max(1, Math.floor(dw * this.q.renderScale));
    const rh = Math.max(1, Math.floor(dh * this.q.renderScale));

    this.displaySize.width = dw;
    this.displaySize.height = dh;
    if (this.screenSize.width === rw && this.screenSize.height === rh && this.hdrRt) return;
    this.screenSize.width = rw;
    this.screenSize.height = rh;

    this.hdrRt?.dispose();
    this.hdrRt = hdrTarget(rw, rh, { depthBuffer: true, name: 'hdr' });
    // The viewmodel gets its own colour+depth buffer with 4x MSAA, cleared to
    // TRANSPARENT black so the composite has real coverage to work with.
    this.viewRt?.dispose();
    this.viewRt = hdrTarget(rw, rh, {
      depthBuffer: true,
      samples: this._viewSamples,
      name: 'viewmodel',
    });
    this.pingRt[0]?.dispose();
    this.pingRt[1]?.dispose();
    this.pingRt[0] = hdrTarget(rw, rh, { name: 'ping0' });
    this.pingRt[1] = hdrTarget(rw, rh, { name: 'ping1' });
    // Only the no-TAA path composites through an LDR intermediate; with TAA on,
    // `fxaa` is null and this was a full-resolution RGBA8 target (13 MB at 3.34
    // MP) allocated on every resize and never sampled once.
    this.ldrRt?.dispose();
    this.ldrRt = null;
    if (this.fxaa) this.ldrRt = new THREE.WebGLRenderTarget(rw, rh, {
      type: THREE.UnsignedByteType,
      format: THREE.RGBAFormat,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      depthBuffer: false,
      stencilBuffer: false,
    });

    this.gbuffer.setSize(rw, rh);
    this.gtao?.setSize(rw, rh);
    this.contact?.setSize(rw, rh);
    this.ssr?.setSize(rw, rh);
    this.taa?.setSize(rw, rh);
    this.motionBlur?.setSize(rw, rh);
    this.dof?.setSize(rw, rh);
    this.bloom?.setSize(rw, rh);

    this.patcher.setScreenSize(rw, rh);
    this.viewComposite.uniforms.uTexel.value.set(1 / rw, 1 / rh);
    this.composite.uniforms.uTexel.value.set(1 / rw, 1 / rh);
    this.composite.uniforms.uResolution.value.set(rw, rh);
    if (this.fxaa) this.fxaa.uniforms.uTexel.value.set(1 / rw, 1 / rh);

    this.depthTexture = this.gbuffer.depthTexture;
    this.velocityTexture = this.gbuffer.velocityTexture;
    this.normalTexture = this.gbuffer.normalTexture;

    for (const p of this.passes) p.resize?.(rw, rh);
    this.taa?.reset();
    this.exposure.reset();
  }

  // ==========================================================================
  //  scene walk
  // ==========================================================================

  _visit(o) {
    if (o.isMesh === true || o.isPoints === true || o.isSprite === true || o.isLine === true) {
      const mat = o.material;
      let transparent = false;
      if (Array.isArray(mat)) {
        for (let i = 0; i < mat.length; i++) {
          this.patcher.patch(mat[i]);
          if (mat[i] && mat[i].transparent === true) transparent = true;
        }
      } else if (mat) {
        this.patcher.patch(mat);
        transparent = mat.transparent === true;
      }

      if (o.isMesh !== true) transparent = true;
      if (o.userData.owProbe !== true) this._foreignMeshes++;

      const ud = o.userData;
      if (transparent || ud.owNoPrepass === true) {
        this._hide[this._nHide++] = o;
      } else {
        this._draw[this._nDraw++] = o;
        if (ud.owNoShadow === true) this._noShadow[this._nNoShadow++] = o;
      }
    } else if (o.isDirectionalLight === true) {
      this._dirLights[this._nDirLights++] = o;
    }
  }

  _collect(scene) {
    this._nDraw = 0;
    this._nHide = 0;
    this._nNoShadow = 0;
    this._nDirLights = 0;
    this._foreignMeshes = 0;
    scene.traverseVisible(this._visit);
  }

  _hideList(list, n) {
    for (let i = 0; i < n; i++) list[i].visible = false;
  }
  _showList(list, n) {
    for (let i = 0; i < n; i++) list[i].visible = true;
  }

  _dirFromLight(light, out) {
    light.updateWorldMatrix(true, false);
    out.setFromMatrixPosition(light.matrixWorld);
    if (light.target) {
      light.target.updateWorldMatrix(true, false);
      this._tmpV3b.setFromMatrixPosition(light.target.matrixWorld);
      out.sub(this._tmpV3b);
    }
    if (out.lengthSq() < 1e-8) out.set(0, 1, 0);
    return out.normalize();
  }

  /** Pick whichever directional light is acting as the sun and take it over. */
  _syncSun(camera) {
    let best = null;
    let bestI = -1;
    for (let i = 0; i < this._nDirLights; i++) {
      const l = this._dirLights[i];
      if (l === this.sun) continue;
      if (l.intensity > bestI) {
        best = l;
        bestI = l.intensity;
      }
    }

    if (best && bestI > 0.01) {
      // Somebody else (the sky) owns the sun now: drop ours and take over its
      // shadowing, because three's single-frustum shadow map cannot compete
      // with cascades.
      if (this.sun.visible) this.sun.visible = false;
      if (best.castShadow) best.castShadow = false;
      this.activeSun = best;
    } else {
      this.sun.visible = true;
      this.activeSun = this.sun;
    }

    this._dirFromLight(this.activeSun, this.sunDir);
    this.sunDirView.copy(this.sunDir).transformDirection(camera.matrixWorldInverse).normalize();
  }

  /**
   * Point the viewmodel's 3-point rig. Directions are authored in VIEW space
   * and rotated into the viewmodel scene every frame, so the weapon's key/fill/
   * rim separation is invariant to where the world sun happens to be.
   */
  _updateViewRig(viewCamera) {
    const s = this.settings;
    // Reference light level for the rig. It has to include the ambient, not just
    // the key: at night the key IS the moon at 0.075, and a rig with an absolute
    // floor would put a glowing white rifle in a moonlit street. Everything in
    // the rig — hemisphere included — is a ratio of this, so autoexposure keeps
    // the weapon at a constant relative brightness at every time of day.
    const ref = Math.max(this.activeSun.intensity, this._ambLevel / 0.15);
    // Sub-linear in the scene level: the meter is exposure-locked after dark, so
    // a rig that tracked the light exactly would put the weapon back in
    // silhouette at night. Every shipped shooter biases the viewmodel up in the
    // dark; gamma 0.65 is that bias, and it is a no-op in full daylight.
    const shaped = REF_DAYLIGHT * Math.pow(Math.min(ref / REF_DAYLIGHT, 1), s.viewKeyGamma);
    const keyI = Math.min(shaped * s.viewKeyScale, s.viewKeyMax);
    this.viewSun.color.copy(this.activeSun.color);
    this.viewSun.intensity = keyI;

    // Fill takes the cool sky hue, rim the warm key hue, so the gun sits in the
    // same light as the street rather than looking like a studio render.
    const h = this._fillHue;
    this.viewKeyFill.color.setRGB(h.x, h.y, h.z);
    this.viewKeyFill.intensity = keyI * s.viewFillRatio;
    const sc = this.activeSun.color;
    this.viewRim.color.setRGB(sc.r, sc.g * 0.94, sc.b * 0.82);
    this.viewRim.intensity = keyI * s.viewRimRatio;
    this.viewFill.intensity = keyI * s.viewHemiRatio;

    // Warm bounce takes the ground-bounce hue the world's own lower fill band
    // uses (_updateBounceFill writes it into _fillHue2), so the gun and gloves
    // pick up the same sand-off-the-street colour the buildings do.
    const g = this._fillHue2;
    if (Math.max(g.x, g.y, g.z) > 1e-5) {
      this.viewBounce.color.setRGB(g.x, g.y * 0.86, g.z * 0.62);
    }
    this.viewBounce.intensity = keyI * s.viewBounceRatio;

    this._placeViewLight(this.viewSun, this._viewKeyDir, viewCamera);
    this._placeViewLight(this.viewKeyFill, this._viewFillDir, viewCamera);
    this._placeViewLight(this.viewRim, this._viewRimDir, viewCamera);
    this._placeViewLight(this.viewBounce, this._viewBounceDir, viewCamera);
  }

  _placeViewLight(light, dirView, viewCamera) {
    const d = this._tmpV3c.copy(dirView).transformDirection(viewCamera.matrixWorld);
    light.target.position.setFromMatrixPosition(viewCamera.matrixWorld);
    light.position.copy(light.target.position).addScaledVector(d, 4);
    light.updateMatrixWorld(true);
    light.target.updateMatrixWorld(true);
  }

  /**
   * Drive the two-band bounce fill from the live sky.
   *
   * The engine has exactly one indirect term — the PMREM sky cubemap — and
   * GTAO multiplies it, so a wall in shade received almost nothing. This adds
   * the two things a real street has: a cool hemisphere of skylight, and the
   * warm light the sunlit side of the street throws back at the shaded side.
   * Both are scaled off the key's intensity so they follow time of day for
   * free, and both are colour-matched to the sky's own published ambient.
   */
  _updateBounceFill() {
    const s = this.settings;
    const u = this.patcher.uniforms;
    const sunI = Math.max(0, this.activeSun.intensity);

    // Hue of the whole-sky band. `sky.ambientColor` is the sky subsystem's own
    // CPU stand-in for it; without a sky system we fall back to the key colour.
    const sky = this.ctx.peek('sky');
    const amb = sky?.ambientColor;
    const hue = this._fillHue;
    if (amb && Math.max(amb.r, amb.g, amb.b) > 1e-5) {
      hue.set(amb.r, amb.g, amb.b);
      // Published whole-sky level, ~15% of the beam. Used as the "how much light
      // is in this scene at all" reference for the viewmodel rig.
      this._ambLevel = Math.max(hue.x, hue.y, hue.z);
    } else {
      hue.set(0.36, 0.56, 1.0);
      this._ambLevel = 0.15 * sunI;
    }
    hue.divideScalar(Math.max(hue.x, hue.y, hue.z));
    // `sky.ambientColor` is the WHOLE-sky average — zenith Rayleigh blue mixed
    // with a horizon band that is nearly achromatic by day and the sunset
    // itself at dusk. This band is not the whole sky: it is what an up-facing
    // or vertical surface sees of the UPPER hemisphere, which is the bluest
    // part of it. Pushing the chroma out from its own luminance recovers that
    // without inventing a hue, and it is what turns a shaded facade from the
    // measured B-R of +0.0002 (4.5% saturation, indistinguishable from grey)
    // into something that reads as skylight.
    {
      const l = 0.2126 * hue.x + 0.7152 * hue.y + 0.0722 * hue.z;
      const k = 1.18;
      hue.set(
        Math.max(0, l + (hue.x - l) * k),
        Math.max(0, l + (hue.y - l) * k),
        Math.max(0, l + (hue.z - l) * k)
      );
      hue.divideScalar(Math.max(hue.x, hue.y, hue.z, 1e-6));
    }

    // The cool band rides the sky's own published irradiance, NOT the key: at
    // night the key is a 0.05 moon and a band scaled off it is nothing, which is
    // how a night frame ends up with a fifth of its pixels under code value 12.
    // `skyFill` stays a fraction of the *beam* by construction, because the sky
    // publishes its ambient as 15% of the beam in daylight.
    const skyRef = this._ambLevel / 0.15;
    const skyLevel = s.skyFill * skyRef;
    u.owSkyFill.value.set(hue.x * skyLevel, hue.y * skyLevel, hue.z * skyLevel);

    // The lower band is sunlight off the road, so it takes the *key's* colour
    // through the ground albedo the sky dome itself uses — warm, not blue.
    const sc = this.activeSun.color;
    const g = this._fillHue2.set(sc.r * 0.33, sc.g * 0.29, sc.b * 0.225);
    g.divideScalar(Math.max(g.x, g.y, g.z, 1e-6));
    const groundLevel = s.groundFill * sunI;
    u.owGroundFill.value.set(g.x * groundLevel, g.y * groundLevel, g.z * groundLevel);

    u.owFillGain.value.set(1, s.bounceFill / Math.max(s.groundFill, 1e-6));
    // The sky publishes an elevation-dependent indirect budget so the key:fill
    // ratio does not invert at golden hour (see SkySystem.indirectScale).
    u.owIndirect.value.x = s.iblDiffuse * (sky?.indirectScale ?? 1);
    u.owIndirect.value.y = s.interiorIndirect;
    this._skyExposureBias = sky?.exposureBias ?? 0;
  }

  /**
   * Publish the coarse interior volumes the indirect gate tests against.
   *
   * The level is authored on one yaw, so a world position reaches level space
   * through a 2D rotation — cheap enough to do per fragment. The volumes are
   * the enterable buildings' own footprints, which the world subsystem already
   * publishes as `world.buildings[].spec`; we do not need per-room geometry
   * because the gate keys off *depth inside the footprint*, and a wall's outer
   * skin is at depth 0 while its inner skin is one thickness in.
   *
   * Runs once, when the world first appears, and then never again.
   */
  _updateRooms() {
    if (this._roomsReady) return;
    const world = this.ctx.peek('world');
    const list = world?.buildings;
    if (!list || !world.levelToWorld) return;
    this._roomsReady = true;

    // Recover the level->world yaw from two transformed level-space points, so
    // this stays correct if the world subsystem re-authors its transform.
    const o = world.levelToWorld(0, 0, 0, this._tmpV3);
    const ox = o.x;
    const oz = o.z;
    const ex = world.levelToWorld(1, 0, 0, this._tmpV3b);
    const c = ex.x - ox;
    const sn = ex.z - oz;
    const inv = 1 / Math.max(1e-6, Math.hypot(c, sn));
    const cs = c * inv;
    const sni = sn * inv;
    // world -> level: p' = R^T (p - o)
    this.patcher.uniforms.owRoomXf.value.set(
      cs,
      sni,
      -(ox * cs + oz * sni),
      -(-ox * sni + oz * cs)
    );

    const rooms = this.patcher.rooms;
    const roomsY = this.patcher.roomsY;
    let n = 0;
    for (const b of list) {
      const sp = b?.spec;
      if (!sp || sp.enterable !== true) continue;
      // A collapsed or ruined shell is open to the sky: it must keep its
      // skylight, or the one room in the level with a hole in its roof is the
      // one that reads as a cave.
      if (sp.collapse === true || sp.ruin === true) continue;
      if (n >= rooms.length) break;
      rooms[n].set(sp.x, sp.z, sp.w * 0.5, sp.d * 0.5);
      // From below the ground slab (so the floor plate counts as interior) up to
      // just under the roof deck — or under a setback, whose terrace is outdoors
      // and sits inside the footprint.
      let top = (b.roofY ?? 12) - 0.06;
      const sb = sp.setback?.from;
      if (sb !== undefined && b.floorY?.[sb] !== undefined) top = b.floorY[sb] - 0.06;
      roomsY[n].set(-0.8, top, 0, 0);
      n++;
    }
    this.patcher.uniforms.owIndirect.value.z = n;
    if (n > 0) console.info(`[render] indirect gate: ${n} interior volumes`);
  }

  _ensureProbe(ctx) {
    // A couple of foreign meshes means another subsystem is still using its
    // own placeholder; the probe only steps aside for a real level.
    const FOREIGN_LIMIT = 6;
    if (this.probeActive) {
      if (this._foreignMeshes >= FOREIGN_LIMIT) {
        ctx.scene.remove(this.probe.group);
        this.probe.dispose();
        this.probeActive = false;
        this.taa?.reset();
      }
      return;
    }
    if (this.frame > 4 || this._foreignMeshes >= FOREIGN_LIMIT) return;
    const g = this.probe.build();
    g.traverse((o) => {
      o.userData.owProbe = true;
    });
    ctx.scene.add(g);
    this.probeActive = true;
  }

  _cullLights(camPos) {
    const s = this.settings;
    for (let i = 0; i < this.lights.length; i++) {
      const e = this.lights[i];
      // If the owner animated the intensity since we last wrote it, adopt the
      // new value as the base rather than fighting them (flickering lamps).
      if (e.applied !== undefined && e.light.intensity !== e.applied) {
        e.baseIntensity = e.light.intensity;
      }
      const d = e.light.position.distanceTo(camPos);
      const fade = 1 - THREE.MathUtils.smoothstep(d, e.range * 0.75, e.range * 1.15);
      // Practicals are held against the sun by the renderer, because the
      // renderer is what owns the key:fill ratio. A "practical" here is a light
      // that asked to be distance-culled inside a room-or-street radius; the FX
      // flash pool deliberately registers at 90 m so the fade never bites it,
      // and a muzzle flash must not be dimmed by a room-lighting control.
      const gain = e.range <= PRACTICAL_RANGE ? s.practicalGain : 1;
      e.applied = e.baseIntensity * fade * gain;
      e.light.intensity = e.applied;
      e.light.visible = fade > 0.002;
    }
  }

  // ==========================================================================
  //  the frame
  // ==========================================================================

  render(ctx) {
    const renderer = this.renderer;
    const { scene, camera, viewScene, viewCamera } = ctx;
    const dt = Math.min(0.1, Math.max(1 / 480, ctx.time.dt || 1 / 60));

    if (this._dynamicResolution) {
      // Only react to plausible gameplay frame times. A long pause caused by
      // tab switching or devtools should not permanently lower image quality.
      const frameMs = dt * 1000;
      if (frameMs >= 4 && frameMs <= 80) {
        this._dynamicFrameMs += (frameMs - this._dynamicFrameMs) * 0.08;
      }
      if (++this._dynamicScaleFrames >= 45) {
        this._dynamicScaleFrames = 0;
        const current = this.q.renderScale;
        let next = current;
        if (this._dynamicFrameMs > 21) next = Math.max(0.48, current - 0.04);
        else if (this._dynamicFrameMs < 14.5) next = Math.min(this._baseRenderScale, current + 0.02);

        if (Math.abs(next - current) >= 0.005) {
          this.q.renderScale = next;
          const canvas = this.ctx.canvas;
          this.resize(
            canvas.clientWidth || globalThis.innerWidth || 1280,
            canvas.clientHeight || globalThis.innerHeight || 720,
            this.ctx
          );
        }
      }
    }

    this.frame++;
    renderer.info.reset();

    camera.updateMatrixWorld();
    viewCamera.updateMatrixWorld();

    this._collect(scene);
    this._ensureProbe(ctx);
    this._syncSun(camera);
    this._updateRooms();
    this._updateBounceFill();
    this._updateViewRig(viewCamera);
    this._camPos.setFromMatrixPosition(camera.matrixWorld);
    this._cullLights(this._camPos);
    this._adsT = this._readAds();

    if (ctx.scene.environment !== this.envMap) {
      // somebody installed a better environment — adopt it everywhere
      this.envMap = ctx.scene.environment;
    }
    // Mirror the world environment onto the viewmodel scene so the weapon is
    // lit by the same IBL, but never stomp an environment the weapons
    // subsystem chose for itself.
    if (
      ctx.viewScene.environment !== ctx.scene.environment &&
      (ctx.viewScene.environment === null ||
        ctx.viewScene.environment === this._assignedViewEnv)
    ) {
      ctx.viewScene.environment = ctx.scene.environment;
      this._assignedViewEnv = ctx.scene.environment;
    }

    // ---- unjittered matrices for velocity + reprojection ------------------
    this._currVP.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    this._invVP.copy(this._currVP).invert();
    if (this._firstFrame) this._prevVP.copy(this._currVP);

    // ---- 2. cascaded shadow maps -----------------------------------------
    const bg = scene.background;
    if (this.csm.enabled) {
      this.csm.update(camera, this.sunDir, this.settings.sunSoftness);
      this.csm.setJitter(this.taa ? this.frame % 8 : 0);
      scene.background = null;
      this._hideList(this._hide, this._nHide);
      this._hideList(this._noShadow, this._nNoShadow);
      this.csm.render(renderer, scene, this._noCascadeCull ? null : this._draw, this._nDraw);
      this._showList(this._noShadow, this._nNoShadow);
      this._showList(this._hide, this._nHide);
      scene.background = bg;
    }

    // ---- 3. TAA jitter ----------------------------------------------------
    // World camera only. The viewmodel is not temporally resolved any more, so
    // jittering its projection would just make it shimmer with nothing to
    // accumulate the offsets back out.
    if (this.taa) this._applyJitter(camera);

    // ---- 4. prepass -------------------------------------------------------
    const gb = this.gbuffer;
    if (this.needsPrepass) {
      scene.background = null;
      this._hideList(this._hide, this._nHide);
      gb.render(renderer, scene, camera, this._currVP, this._prevVP, true);
      this._showList(this._hide, this._nHide);
      scene.background = bg;
    }

    const feat = this.patcher.uniforms.owFeat.value;
    feat.set(0, 0, 0, 1);

    // ---- 5/6/7. AO, contact shadows, reflections --------------------------
    if (this.gtao && this.needsPrepass) {
      this.patcher.uniforms.owAoTex.value = this.gtao.render(
        renderer,
        gb,
        camera,
        this.frame,
        !!this.taa
      );
      this.aoTexture = this.patcher.uniforms.owAoTex.value;
      feat.x = 1;
    }
    if (this.contact && this.needsPrepass) {
      this.patcher.uniforms.owContactTex.value = this.contact.render(
        renderer,
        gb,
        camera,
        this.sunDirView,
        this.frame
      );
      feat.y = 1;
    }
    if (this.ssr && this.needsPrepass && !this._firstFrame) {
      // Previous frame's resolved colour. Without TAA the HDR target still
      // holds last frame at this point in the schedule, which is exactly what
      // we want to reflect.
      const src = this.taa ? this.taa.previousTexture : this.hdrRt.texture;
      this.patcher.uniforms.owSsrTex.value = this.ssr.render(renderer, gb, src, camera, this.frame);
      feat.z = 1;
    }

    // ---- 8. forward world pass -------------------------------------------
    this.csm.uniforms.owSunDirView.value.copy(this.sunDirView);
    renderer.setRenderTarget(this.hdrRt);
    renderer.clear(true, true, false);
    renderer.render(scene, camera);

    // ---- 9. viewmodel into its OWN colour+depth target --------------------
    // NOT into the world buffer: see the header note. The gbuffer is left
    // describing the world alone, which is what TAA, motion blur, the ADS DOF
    // and the volumetric fog all need it to be.
    this._viewVisible = viewScene.children.length > this._viewRigChildren;
    if (this._viewVisible) {
      // The viewmodel is only in the cascades if its camera shares the world
      // camera's position; otherwise the world-space lookup would be nonsense.
      this._tmpV3.setFromMatrixPosition(viewCamera.matrixWorld);
      const coherent = this._tmpV3.distanceToSquared(this._camPos) < 0.25;
      const prevStrength = this.csm.uniforms.owCsmParams.value.x;
      const prevFeat = feat.y;
      if (!coherent) this.csm.uniforms.owCsmParams.value.x = 0;
      feat.y = 0; // contact shadows are a world-space buffer; not for the gun
      this.csm.uniforms.owSunDirView.value
        .copy(this.sunDir)
        .transformDirection(viewCamera.matrixWorldInverse)
        .normalize();

      // A weapon at the shoulder sees maybe half the sky — the shooter's own
      // head, chest and arms take the rest — so the hemispheric fill is
      // occluded for the viewmodel exactly the way its envMapIntensity is.
      // Without this the gun floats in more indirect light than the street.
      const uSky = this.patcher.uniforms.owSkyFill.value;
      const uGnd = this.patcher.uniforms.owGroundFill.value;
      this._fillSkySave.copy(uSky);
      this._fillGroundSave.copy(uGnd);
      uSky.multiplyScalar(this.settings.viewFillOcclusion);
      uGnd.multiplyScalar(this.settings.viewFillOcclusion);
      // The interior gate is a WORLD-space volume test and the viewmodel's
      // world position is the camera's, so standing in a shop would drop the
      // weapon's whole indirect term at once. The gun has its own rig; skip it.
      const roomN = this.patcher.uniforms.owIndirect.value.z;
      this.patcher.uniforms.owIndirect.value.z = 0;

      // Still walked every frame — that is where new weapon/hand materials get
      // the shadow/AO/fill injection. It just no longer feeds the gbuffer.
      this._collectViewScene(viewScene);

      renderer.setRenderTarget(this.viewRt);
      // Transparent clear: the composite needs coverage, and the MSAA resolve
      // turns partially covered edge pixels into premultiplied fractional alpha.
      renderer.setClearColor(0x000000, 0);
      renderer.clear(true, true, false);
      renderer.render(viewScene, viewCamera);
      renderer.setClearColor(0x000000, 1);

      this.csm.uniforms.owCsmParams.value.x = prevStrength;
      feat.y = prevFeat;
      uSky.copy(this._fillSkySave);
      uGnd.copy(this._fillGroundSave);
      this.patcher.uniforms.owIndirect.value.z = roomN;
    }

    if (this.taa) this._removeJitter(camera);

    // ---- 10. TAA ----------------------------------------------------------
    let color = this.hdrRt.texture;
    if (this.taa) {
      color = this.taa.render(renderer, color, gb, this._invVP, this._prevVP);
    }

    // ---- 11. motion blur --------------------------------------------------
    if (this.motionBlur) {
      const shutter = this.settings.shutter * (1 / 60 / dt);
      color = this.motionBlur.render(renderer, color, gb, this.frame, shutter);
    }

    // ---- 12. ADS depth of field ------------------------------------------
    // World only, and only while the sights are actually up. The viewmodel is
    // composited afterwards, so the optic body and the reticle stay sharp by
    // construction rather than by masking.
    if (this.dof && this._adsT > 0.01 && this.needsPrepass) {
      const dofOut = this.pingRt[this._pingIndex];
      color = this.dof.render(
        renderer,
        color,
        gb,
        dofOut,
        this._adsT,
        this.settings,
        this.frame
      );
      this._pingIndex ^= 1;
    }

    // ---- 13. registered passes -------------------------------------------
    for (let i = 0; i < this.passes.length; i++) {
      const p = this.passes[i];
      if (p.enabled === false) continue;
      const out = this.pingRt[this._pingIndex];
      p.render(renderer, color, out, this);
      color = out.texture;
      this._pingIndex ^= 1;
    }

    // ---- 14. viewmodel composite -----------------------------------------
    // After the registered passes on purpose: the volumetric fog and haze pass
    // are depth-driven and the gbuffer now holds the WORLD depth at the gun's
    // pixels, so compositing earlier would bury the weapon in 40 m of aerial
    // perspective. Before metering and bloom, so the muzzle flash still meters
    // and still blooms.
    if (this._viewVisible) {
      const vu = this.viewComposite.uniforms;
      const out = this.pingRt[this._pingIndex];
      vu.tColor.value = color;
      vu.tView.value = this.viewRt.texture;
      this.viewComposite.render(renderer, out);
      color = out.texture;
      this._pingIndex ^= 1;
    }

    // ---- 15. metering -----------------------------------------------------
    const s = this.settings;
    const exposureTex = this.exposure.update(
      renderer,
      color,
      this.screenSize.width,
      this.screenSize.height,
      s.autoExposure ? dt : 1e3,
      // The sky publishes a metering compensation for the current sun elevation:
      // a street canyon under a four-degree sun is entirely in shade, and a meter
      // weighted onto that geometry opens up two stops and flattens the sky it is
      // lit by. See SkySystem.exposureBias.
      s.exposureBias + this._skyExposureBias,
      s.exposureKey,
      this.needsPrepass ? this.depthTexture : null
    );
    this.exposureTexture = exposureTex;

    // ---- 16. bloom --------------------------------------------------------
    let bloomTex = null;
    if (this.bloom) {
      bloomTex = this.bloom.render(
        renderer,
        color,
        this.screenSize.width,
        this.screenSize.height,
        exposureTex
      );
    }

    // ---- 17/18. composite -------------------------------------------------
    const cu = this.composite.uniforms;
    cu.tColor.value = color;
    cu.tBloom.value = bloomTex ?? color;
    cu.tExposure.value = exposureTex;
    cu.uGrade.value.x = bloomTex ? s.bloomStrength : 0;
    cu.uGrade.value.z = this.taa ? s.sharpen : 0;
    // Vignette closes in with the sight picture.
    cu.uLens.value.y = s.vignette + (s.adsVignette - s.vignette) * this._adsT;
    cu.uLens.value.w = ctx.time.elapsed;
    cu.uLook.value.w = this.ctx.config.exposure ?? 1;

    if (this.debugView) {
      this._renderDebug(renderer, color);
    } else if (this.fxaa) {
      this.composite.render(renderer, this.ldrRt);
      this.fxaa.uniforms.tColor.value = this.ldrRt.texture;
      this.fxaa.render(renderer, null);
    } else {
      this.composite.render(renderer, null);
    }

    // ---- bookkeeping ------------------------------------------------------
    // Only world objects are in the gbuffer now, so only their transforms need
    // remembering for next frame's velocity.
    gb.beginRecord();
    gb.recordMatrices(this._draw, this._nDraw);
    gb.endRecord();
    this._prevVP.copy(this._currVP);
    this._firstFrame = false;
    renderer.setRenderTarget(null);

    if (this._probeExposure) this._logExposure();
  }

  /**
   * The viewmodel scene walk. Its only job now is material patching — the
   * viewmodel is no longer in the gbuffer, so nothing needs its transform
   * remembered for a velocity difference.
   */
  _visitView(o) {
    if (o.isMesh === true) {
      const m = o.material;
      if (Array.isArray(m)) for (let i = 0; i < m.length; i++) this.patcher.patch(m[i]);
      else if (m) this.patcher.patch(m);
    }
  }

  /**
   * Dev aid: show an intermediate buffer full-screen.
   * `ctx.get('render').debugView = 'ao' | 'normal' | 'velocity' | 'depth' |
   *  'ssr' | 'ssrmask' | 'contact' | 'bloom' | null`
   */
  _renderDebug(renderer, color) {
    if (!this._debugPass) this._debugPass = createDebug();
    const u = this._debugPass.uniforms;
    const gb = this.gbuffer;
    const map = {
      ao: [this.aoTexture, 0],
      normal: [gb.normalTexture, 1],
      velocity: [gb.velocityTexture, 2],
      depth: [gb.depthTexture, 3],
      ssr: [this.ssr?.texture, 4],
      ssrmask: [this.ssr?.texture, 5],
      contact: [this.contact?.texture, 0],
      bloom: [this.bloom?.texture, 4],
      view: [this.viewRt?.texture, 4],
      viewalpha: [this.viewRt?.texture, 5],
      color: [color, 4],
    };
    const entry = map[this.debugView] ?? map.color;
    u.tSrc.value = entry[0] ?? color;
    u.uMode.value = entry[1];
    this._debugPass.render(renderer, null);
  }

  /** Dev aid: dump the metering chain. `render.debugExposure()` in console. */
  debugExposure() {
    const buf = this._readback;
    this.renderer.readRenderTargetPixels(this.exposure.rt1, 0, 0, 1, 1, buf);
    const avgLog = buf[0] / Math.max(buf[1], 1e-4);
    const out = this._readback2;
    this.renderer.readRenderTargetPixels(
      this.exposure.adapt[this.exposure._flip],
      0,
      0,
      1,
      1,
      out
    );
    return { avgLum: Math.pow(2, avgLog), ev100: out[1], exposure: out[0] };
  }

  _logExposure() {
    if (this.frame % 90 !== 0) return;
    const d = this.debugExposure();
    const u = this.patcher.uniforms;
    const v3 = (x) => `${x.x.toFixed(3)},${x.y.toFixed(3)},${x.z.toFixed(3)}`;
    console.info(
      `[render] frame ${this.frame} avgLum ${d.avgLum.toFixed(4)} ev100 ${d.ev100.toFixed(2)} exposure ${d.exposure.toFixed(4)} ` +
        `sun=${this.activeSun.intensity.toFixed(3)} skyFill=${v3(u.owSkyFill.value)} gndFill=${v3(u.owGroundFill.value)} ` +
        `ibl=${u.owIndirect.value.x.toFixed(3)} indoor=${u.owIndirect.value.y.toFixed(3)} rooms=${u.owIndirect.value.z}`
    );
  }

  _collectViewScene(viewScene) {
    viewScene.traverseVisible(this._visitView);
  }

  /**
   * Sub-pixel offset for the TAA accumulation. WORLD camera only — the
   * viewmodel has its own MSAA target and no temporal history, so a jitter on
   * `viewCamera` would be a permanent sub-pixel wobble with nothing to resolve
   * it back out.
   */
  _applyJitter(camera) {
    const j = this.taa.nextJitter();
    const jx = (j.x * 2) / this.screenSize.width;
    const jy = (j.y * 2) / this.screenSize.height;
    this._jitterSaved.set(camera.projectionMatrix.elements[8], camera.projectionMatrix.elements[9]);
    camera.projectionMatrix.elements[8] += jx;
    camera.projectionMatrix.elements[9] += jy;
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
    this._jittered = true;
  }

  _removeJitter(camera) {
    if (!this._jittered) return;
    camera.projectionMatrix.elements[8] = this._jitterSaved.x;
    camera.projectionMatrix.elements[9] = this._jitterSaved.y;
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
    this._jittered = false;
  }

  /**
   * ADS engagement, 0..1. The weapons subsystem owns the transition; we only
   * read it, and never require it to exist.
   */
  _readAds() {
    if (!this._weapons) this._weapons = this.ctx.peek('weapons') || null;
    const w = this._weapons;
    if (!w) return 0;
    const t = w.adsProgress;
    return typeof t === 'number' && t === t ? Math.min(1, Math.max(0, t)) : 0;
  }

  dispose() {
    this.csm.dispose();
    this.gbuffer.dispose();
    this.gtao?.dispose();
    this.contact?.dispose();
    this.ssr?.dispose();
    this.taa?.dispose();
    this.motionBlur?.dispose();
    this.dof?.dispose();
    this.bloom?.dispose();
    this.exposure.dispose();
    this.composite.dispose();
    this.viewComposite.dispose();
    this.fxaa?.dispose();
    this.lut.texture.dispose();
    this.envEquirect?.dispose();
    this.hdrRt?.dispose();
    this.viewRt?.dispose();
    this.ldrRt?.dispose();
    this.pingRt[0]?.dispose();
    this.pingRt[1]?.dispose();
    this.envTarget?.dispose();
    if (this.probeActive) {
      this.ctx?.scene.remove(this.probe.group);
      this.probe.dispose();
    }
    this._debugPass?.dispose();
    this.patcher.dispose();
    this.renderer.dispose();
  }
}
