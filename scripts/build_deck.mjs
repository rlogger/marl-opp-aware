import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

async function loadArtifactTool() {
  try {
    return await import("@oai/artifact-tool");
  } catch (error) {
    if (!process.env.ARTIFACT_TOOL_WORKSPACE) {
      throw error;
    }
    const modulePath = path.join(
      process.env.ARTIFACT_TOOL_WORKSPACE,
      "node_modules/@oai/artifact-tool/dist/artifact_tool.mjs",
    );
    return import(pathToFileURL(modulePath).href);
  }
}

const { Presentation, PresentationFile } = await loadArtifactTool();

const __filename = fileURLToPath(import.meta.url);
const ROOT = path.dirname(path.dirname(__filename));
const PLOTS = path.join(ROOT, "plots");
const FINAL_PPTX = process.env.FINAL_PPTX
  ? path.resolve(process.env.FINAL_PPTX)
  : path.join(ROOT, "marl_opp_aware_results.pptx");
const SCRATCH = process.env.ARTIFACT_TOOL_WORKSPACE || path.join(ROOT, ".deck-work");
const QA_DIR = process.env.DECK_QA_DIR || path.join(SCRATCH, "qa");

const W = 1280;
const H = 720;
const INK = "#0A0A0A";
const MUTED = "#4F5661";
const FAINT = "#E6E8EC";
const PANEL = "#F4F5F7";
const CANVAS = "#FFFFFF";
const ACCENT = "#0D6E7A";
const WARNING = "#C84C2A";

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

function contentType(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  if (ext === ".jpg" || ext === ".jpeg") return "image/jpeg";
  if (ext === ".gif") return "image/gif";
  if (ext === ".webp") return "image/webp";
  return "image/png";
}

function addText(slide, text, position, style = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontSize: 22,
    color: INK,
    ...style,
  };
  return shape;
}

function addPanel(slide, position, fill = PANEL) {
  return slide.shapes.add({
    geometry: "rect",
    position,
    fill,
    line: { style: "solid", fill, width: 0 },
  });
}

function addRule(slide, position, fill = ACCENT) {
  return slide.shapes.add({
    geometry: "rect",
    position,
    fill,
    line: { style: "solid", fill, width: 0 },
  });
}

function addTitle(slide, title, kicker = "") {
  if (kicker) {
    addText(slide, kicker.toUpperCase(), { left: 56, top: 34, width: 720, height: 24 }, {
      fontSize: 13,
      bold: true,
      color: MUTED,
    });
  }
  addText(slide, title, { left: 56, top: 62, width: 1080, height: 96 }, {
    fontSize: 42,
    bold: true,
    color: INK,
  });
}

function addFooter(slide, n) {
  addText(slide, String(n).padStart(2, "0"), { left: 1184, top: 658, width: 48, height: 24 }, {
    fontSize: 14,
    color: MUTED,
    alignment: "right",
  });
}

function addMetric(slide, value, label, position, color = ACCENT) {
  addText(slide, value, position, { fontSize: 54, bold: true, color });
  addText(slide, label, {
    left: position.left,
    top: position.top + 66,
    width: position.width,
    height: 54,
  }, { fontSize: 18, color: MUTED });
}

function addBullets(slide, items, position, style = {}) {
  addText(slide, items.map((item) => `- ${item}`).join("\n"), position, {
    fontSize: 22,
    color: MUTED,
    ...style,
  });
}

async function addPlot(slide, fileName, position, alt, fit = "contain") {
  const filePath = path.join(PLOTS, fileName);
  const bytes = await fs.readFile(filePath);
  slide.images.add({
    blob: bytes,
    contentType: contentType(filePath),
    alt,
    fit,
    position,
  });
}

function addNotes(slide, notes) {
  slide.speakerNotes.textFrame.setText(notes);
  slide.speakerNotes.setVisible(true);
}

async function addPlotSlide(presentation, n, cfg) {
  const slide = presentation.slides.add();
  slide.background.fill = CANVAS;
  addTitle(slide, cfg.title, cfg.kicker);
  addFooter(slide, n);

  addPanel(slide, cfg.plotFrame);
  await addPlot(slide, cfg.plot, {
    left: cfg.plotFrame.left + 16,
    top: cfg.plotFrame.top + 16,
    width: cfg.plotFrame.width - 32,
    height: cfg.plotFrame.height - 32,
  }, cfg.alt);

  addRule(slide, { left: cfg.textFrame.left, top: cfg.textFrame.top, width: 6, height: cfg.textFrame.height });
  addBullets(slide, cfg.bullets, {
    left: cfg.textFrame.left + 22,
    top: cfg.textFrame.top - 2,
    width: cfg.textFrame.width - 22,
    height: cfg.textFrame.height,
  }, { fontSize: cfg.fontSize || 22 });

  if (cfg.callout) {
    addText(slide, cfg.callout, cfg.calloutFrame, {
      fontSize: 26,
      bold: true,
      color: INK,
    });
  }

  addNotes(slide, cfg.notes);
}

async function main() {
  await fs.mkdir(QA_DIR, { recursive: true });
  const presentation = Presentation.create({ slideSize: { width: W, height: H } });

  {
    const slide = presentation.slides.add();
    slide.background.fill = CANVAS;
    addText(slide, "Adaptive Opponent\nModeling for MARL", {
      left: 56,
      top: 78,
      width: 830,
      height: 170,
    }, { fontSize: 64, bold: true, color: INK });
    addText(slide, "A controlled study of hidden opponent intent, calibrated belief, and belief-conditioned planning.", {
      left: 56,
      top: 278,
      width: 850,
      height: 70,
    }, { fontSize: 26, color: MUTED });
    addPanel(slide, { left: 56, top: 446, width: 1168, height: 144 });
    addMetric(slide, "4.31", "captures/episode with inferred-belief planning", {
      left: 88,
      top: 474,
      width: 320,
      height: 64,
    });
    addMetric(slide, "-47%", "wrong hard intent vs. opponent-blind policy", {
      left: 468,
      top: 474,
      width: 330,
      height: 64,
    }, WARNING);
    addMetric(slide, "0 labels", "JEPA belief recovers the intent signal", {
      left: 850,
      top: 474,
      width: 300,
      height: 64,
    }, INK);
    addFooter(slide, 1);
    addNotes(slide, "Open with the one-sentence thesis: infer the opponent's hidden strategy as a calibrated belief, then plan against that belief. Be precise that this is a controlled MARL study, not yet a broad SOTA model-based MARL claim. The deck is intentionally minimal; the defense lives in the speaker notes and repo artifacts.");
  }

  {
    const slide = presentation.slides.add();
    slide.background.fill = CANVAS;
    addTitle(slide, "The experiment isolates one question", "Setup");
    addText(slide, "Can a predator team use early opponent behavior to plan better than a team that reacts to a point estimate?", {
      left: 56,
      top: 178,
      width: 760,
      height: 86,
    }, { fontSize: 30, bold: true, color: INK });
    const rows = [
      ["Hidden state", "The prey draws a corner intent each episode; predators never observe it directly."],
      ["Evidence", "The first k prey steps become a posterior belief over four intents."],
      ["Planner", "Predators sample that belief inside a short Monte-Carlo lookahead."],
      ["Claim boundary", "The opponent is static within an episode; adaptive or switching prey is next."],
    ];
    let y = 308;
    for (const [label, body] of rows) {
      addRule(slide, { left: 56, top: y + 10, width: 72, height: 4 }, label === "Claim boundary" ? WARNING : ACCENT);
      addText(slide, label, { left: 154, top: y - 4, width: 220, height: 34 }, {
        fontSize: 22,
        bold: true,
        color: INK,
      });
      addText(slide, body, { left: 398, top: y - 4, width: 720, height: 54 }, {
        fontSize: 22,
        color: MUTED,
      });
      y += 76;
    }
    addFooter(slide, 2);
    addNotes(slide, "This slide prevents overclaiming. The environment is deliberately controlled: same map, hidden per-episode strategy, and measurable ground truth. The value is causal clarity: if performance moves, we know whether it came from inference, uncertainty, or lookahead.");
  }

  await addPlotSlide(presentation, 3, {
    title: "The belief becomes useful evidence quickly",
    kicker: "Part 1",
    plot: "part2_intent_eval.png",
    alt: "Intent inference and capture ladder for hidden-intent predator-prey.",
    plotFrame: { left: 56, top: 166, width: 710, height: 432 },
    textFrame: { left: 808, top: 184, width: 360, height: 226 },
    bullets: [
      "Accuracy rises from 0.37 at 3 steps to 0.97 at 25 steps.",
      "Entropy falls from 1.35 to 0.03 nats.",
      "A wrong hard guess drops captures from 2.68 to 1.42.",
    ],
    callout: "Uncertainty is not decoration. It is the difference between useful evidence and a brittle guess.",
    calloutFrame: { left: 808, top: 468, width: 360, height: 94 },
    notes: "Read the plot as two claims. First, intent is inferable from behavior. Second, compressing the posterior into the wrong point estimate is actively harmful. That is why the planner should consume the belief distribution, not just argmax intent.",
  });

  await addPlotSlide(presentation, 4, {
    title: "Planning over the belief is the core result",
    kicker: "Part 2",
    plot: "part2_planner.png",
    alt: "Planner comparison for opponent-blind, flat belief, oracle reactive, and inferred-belief planning.",
    plotFrame: { left: 56, top: 166, width: 690, height: 432 },
    textFrame: { left: 790, top: 184, width: 382, height: 220 },
    bullets: [
      "Flat-belief planner: 3.07 captures/episode.",
      "Reactive oracle with true intent: 4.05.",
      "Planner + inferred belief: 4.31.",
    ],
    callout: "The result is not just having a model. It is sampling a calibrated opponent belief inside lookahead.",
    calloutFrame: { left: 790, top: 466, width: 394, height: 96 },
    notes: "This is the slide to linger on. The fair phrasing is that inferred-belief planning beats the flat-belief ablation and matches or slightly exceeds a reactive oracle. Do not say it beats an oracle planner; that should be run before submission.",
  });

  await addPlotSlide(presentation, 5, {
    title: "JEPA makes the belief label-free",
    kicker: "Representation",
    plot: "jepa_vs_vae_encoder.png",
    alt: "JEPA and VAE comparison for self-supervised opponent intent representation.",
    plotFrame: { left: 56, top: 166, width: 708, height: 432 },
    textFrame: { left: 808, top: 184, width: 360, height: 220 },
    bullets: [
      "Same trajectories, same 2-D latent, no strategy labels.",
      "JEPA probe: 0.89 vs VAE: 0.53.",
      "ARI: 0.65 vs 0.14.",
      "JEPA belief planner reaches 4.08 captures/episode.",
    ],
    callout: "Predicting future behavior keeps the useful intent signal and drops evasion noise.",
    calloutFrame: { left: 808, top: 482, width: 370, height: 80 },
    notes: "This is the literature bridge: JEPA-style predictive learning is a better fit than reconstruction when the important variable is the stable cause of future behavior. It gives the paper a label-free path while preserving the central belief-planning story.",
  });

  await addPlotSlide(presentation, 6, {
    title: "The main caveat is dynamics, not representation",
    kicker: "Boundary",
    plot: "part4_jepa_world_planner.png",
    alt: "JEPA world-model planning result showing poor learned-dynamics performance.",
    plotFrame: { left: 56, top: 166, width: 690, height: 432 },
    textFrame: { left: 790, top: 184, width: 386, height: 236 },
    bullets: [
      "Self-supervised belief works in the simulator planner.",
      "The learned latent world planner reaches only 0.57 captures.",
      "So the paper should claim opponent belief and planning, not solved learned dynamics.",
    ],
    callout: "The negative result is useful: it tells reviewers exactly where the next engineering risk is.",
    calloutFrame: { left: 790, top: 478, width: 386, height: 84 },
    notes: "Use this slide to earn trust. The representation is strong enough to guide planning, but the learned dynamics model is not reliable enough yet. That makes the submission story sharper, not weaker, because it separates representation evidence from model-based control evidence.",
  });

  {
    const slide = presentation.slides.add();
    slide.background.fill = CANVAS;
    addTitle(slide, "The BC follow-up says strategy information has value", "Secondary evidence");
    addPanel(slide, { left: 56, top: 166, width: 532, height: 362 });
    addPanel(slide, { left: 652, top: 166, width: 532, height: 362 });
    await addPlot(slide, "mopa_bc_vs_mappo.png", {
      left: 72,
      top: 182,
      width: 500,
      height: 330,
    }, "Behavior cloning capture comparison against MAPPO expert and random policy.");
    await addPlot(slide, "mopa_bc_latent_deploy.png", {
      left: 668,
      top: 182,
      width: 500,
      height: 330,
    }, "Deployment comparison for placement-blind, latent-conditioned, and oracle-placement BC.");
    addBullets(slide, [
      "BC recovers most of the expert edge: 1.22 captures vs 1.35 expert and 0.40 random.",
      "Oracle placement improves deployed captures; the current unsupervised latent gain is modest.",
    ], { left: 88, top: 556, width: 1000, height: 72 }, { fontSize: 21 });
    addFooter(slide, 7);
    addNotes(slide, "This slide is supporting evidence, not the main claim. The clone is good enough that imitation is not the bottleneck. True strategy information helps, but the unsupervised resource-axis latent is not yet a clean paper-level win.");
  }

  {
    const slide = presentation.slides.add();
    slide.background.fill = CANVAS;
    addTitle(slide, "What to say, and what not to overclaim", "Close");
    const blocks = [
      {
        x: 56,
        title: "Claim now",
        color: ACCENT,
        body: "In a controlled hidden-intent MARL game, calibrated opponent belief plus lookahead is stronger than blind reaction or hard guessed intent.",
      },
      {
        x: 456,
        title: "Say carefully",
        color: WARNING,
        body: "The opponent is static within each episode, the strongest planner uses simulator dynamics, and the oracle comparison is reactive.",
      },
      {
        x: 856,
        title: "Run next",
        color: INK,
        body: "Adaptive or switching opponents; oracle planner; HOP, MAZero, MAMBA, MARIE, MATWM, MBOM, and AORPO baselines.",
      },
    ];
    for (const block of blocks) {
      addRule(slide, { left: block.x, top: 188, width: 84, height: 6 }, block.color);
      addText(slide, block.title, { left: block.x, top: 214, width: 300, height: 40 }, {
        fontSize: 26,
        bold: true,
        color: INK,
      });
      addText(slide, block.body, { left: block.x, top: 272, width: 310, height: 230 }, {
        fontSize: 24,
        color: MUTED,
      });
    }
    addText(slide, "A strong paper pitch is not \"we solved MARL.\" It is \"we isolated the value of calibrated opponent belief, then showed why planning should consume the whole belief.\"", {
      left: 56,
      top: 574,
      width: 1050,
      height: 58,
    }, { fontSize: 25, bold: true, color: INK });
    addFooter(slide, 8);
    addNotes(slide, "Close by tying the paper to the literature. HOP is the closest belief-plus-planning comparison. MAZero, MAMBA, MARIE, and MATWM cover model-based MARL. MBOM and AORPO cover opponent modeling. The repo is now packaged like a paper artifact; the remaining risk is scientific comparison breadth.");
  }

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(path.join(QA_DIR, `${stem}.png`), await presentation.export({ slide, format: "png", scale: 1 }));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(QA_DIR, `${stem}.layout.json`), await layout.text());
  }

  await writeBlob(path.join(QA_DIR, "deck-montage.webp"), await presentation.export({
    format: "webp",
    montage: true,
    scale: 1,
  }));

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(FINAL_PPTX);
  console.log(`Wrote ${FINAL_PPTX}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
