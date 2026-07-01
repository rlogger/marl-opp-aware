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
const PANEL = "#F4F5F7";
const CANVAS = "#FFFFFF";
const ACCENT = "#0D6E7A";
const WARNING = "#C84C2A";
const RULE = "#D8DDE3";

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
  addText(slide, title, { left: 56, top: 62, width: 1080, height: 98 }, {
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

function addBullets(slide, items, position, style = {}) {
  addText(slide, items.map((item) => `- ${item}`).join("\n"), position, {
    fontSize: 22,
    color: MUTED,
    ...style,
  });
}

function addNumber(slide, value, label, position, color = ACCENT) {
  addText(slide, value, position, { fontSize: 56, bold: true, color });
  addText(slide, label, {
    left: position.left,
    top: position.top + 70,
    width: position.width,
    height: 70,
  }, { fontSize: 19, color: MUTED });
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
      fontSize: 25,
      bold: true,
      color: INK,
    });
  }
  addNotes(slide, cfg.notes);
}

function addColumnBlock(slide, cfg) {
  addRule(slide, { left: cfg.x, top: cfg.y, width: 84, height: 6 }, cfg.color || ACCENT);
  addText(slide, cfg.title, { left: cfg.x, top: cfg.y + 28, width: cfg.width, height: 44 }, {
    fontSize: 25,
    bold: true,
    color: INK,
  });
  addText(slide, cfg.body, { left: cfg.x, top: cfg.y + 86, width: cfg.width, height: cfg.height || 210 }, {
    fontSize: cfg.fontSize || 22,
    color: MUTED,
  });
}

async function main() {
  await fs.mkdir(QA_DIR, { recursive: true });
  const presentation = Presentation.create({ slideSize: { width: W, height: H } });

  {
    const slide = presentation.slides.add();
    slide.background.fill = CANVAS;
    addText(slide, "Adaptive Opponent\nModeling for MARL", {
      left: 150,
      top: 240,
      width: 980,
      height: 170,
    }, {
      fontSize: 64,
      bold: true,
      color: INK,
      alignment: "center",
    });
    addNotes(slide, "Open slowly. The whole talk is about one controlled question: can a team measure an opponent's hidden strategy, keep uncertainty in the loop, and use that belief during planning?");
  }

  {
    const slide = presentation.slides.add();
    slide.background.fill = CANVAS;
    addTitle(slide, "The paper claim is narrow enough to defend", "Claim");
    addText(slide, "A predator team performs better when it plans against a calibrated belief over the prey's hidden intent.", {
      left: 56,
      top: 182,
      width: 860,
      height: 84,
    }, { fontSize: 32, bold: true, color: INK });
    addNumber(slide, "4.31", "captures/episode with inferred-belief planning", {
      left: 72,
      top: 364,
      width: 310,
      height: 70,
    });
    addNumber(slide, "1.42", "captures/episode when the model uses a wrong hard intent", {
      left: 476,
      top: 364,
      width: 330,
      height: 70,
    }, WARNING);
    addNumber(slide, "4.08", "captures/episode with the label-free JEPA belief", {
      left: 888,
      top: 364,
      width: 300,
      height: 70,
    }, INK);
    addFooter(slide, 2);
    addNotes(slide, "Use this slide to state the exact claim. The evidence comes from captures, not action accuracy. The 4.31 result is the main positive result. The 1.42 result explains why hard labels can hurt control.");
  }

  {
    const slide = presentation.slides.add();
    slide.background.fill = CANVAS;
    addTitle(slide, "The task hides strategy inside the prey", "Setup");
    const rows = [
      ["Episode draw", "The prey receives one hidden corner intent at reset."],
      ["Observation", "Predators see motion, positions, obstacles, and rewards."],
      ["Belief", "The first k prey steps produce a posterior over four intents."],
      ["Control", "The planner samples that posterior during lookahead."],
    ];
    let y = 188;
    for (const [label, body] of rows) {
      addRule(slide, { left: 72, top: y + 16, width: 86, height: 4 }, ACCENT);
      addText(slide, label, { left: 190, top: y, width: 240, height: 40 }, {
        fontSize: 24,
        bold: true,
        color: INK,
      });
      addText(slide, body, { left: 460, top: y, width: 660, height: 46 }, {
        fontSize: 24,
        color: MUTED,
      });
      y += 88;
    }
    addText(slide, "This setup keeps the map fixed. Any lift must come from reading behavior and using the belief.", {
      left: 72,
      top: 582,
      width: 930,
      height: 44,
    }, { fontSize: 24, bold: true, color: INK });
    addFooter(slide, 3);
    addNotes(slide, "Explain the lived detail: three blue predators chase one red prey in the JaxMARL simple-tag arena. The prey knows the intent. The predators infer it from motion. This design isolates opponent modeling from map perception.");
  }

  await addPlotSlide(presentation, 4, {
    title: "The belief signal becomes measurable",
    kicker: "Part 1",
    plot: "part2_intent_eval.png",
    alt: "Intent inference and capture ladder for hidden-intent predator-prey.",
    plotFrame: { left: 56, top: 166, width: 704, height: 432 },
    textFrame: { left: 800, top: 184, width: 370, height: 210 },
    bullets: [
      "Accuracy rises from 0.37 at 3 observed steps to 0.97 at 25.",
      "Posterior entropy falls from 1.35 to 0.03 nats.",
      "The belief sharpens before the episode ends.",
    ],
    callout: "This validates the Part 1 encoder: early motion carries the hidden intent.",
    calloutFrame: { left: 800, top: 474, width: 372, height: 86 },
    notes: "Read the plot from left to right. The encoder starts uncertain, then locks onto the intent as the prey commits. This matters because the planner can act before the final steps, when interception still has value.",
  });

  {
    const slide = presentation.slides.add();
    slide.background.fill = CANVAS;
    addTitle(slide, "Hard intent labels can damage control", "Ablation");
    addPanel(slide, { left: 72, top: 190, width: 326, height: 242 });
    addPanel(slide, { left: 476, top: 190, width: 326, height: 242 });
    addPanel(slide, { left: 880, top: 190, width: 326, height: 242 });
    addNumber(slide, "2.68", "opponent-blind baseline", { left: 104, top: 228, width: 250, height: 70 });
    addNumber(slide, "2.56", "hard intent inferred at k=8", { left: 508, top: 228, width: 250, height: 70 }, INK);
    addNumber(slide, "1.42", "confident wrong intent", { left: 912, top: 228, width: 250, height: 70 }, WARNING);
    addRule(slide, { left: 86, top: 516, width: 7, height: 74 }, WARNING);
    addText(slide, "A point estimate throws away uncertainty. When the estimate is wrong, the predators move toward the wrong interception.", {
      left: 112,
      top: 508,
      width: 920,
      height: 78,
    }, { fontSize: 27, bold: true, color: INK });
    addFooter(slide, 5);
    addNotes(slide, "This slide explains why the method carries a distribution. The wrong hard-intent condition is a stress test. It measured 1.42 captures, far below 2.68 for the blind policy. That failure supports uncertainty-aware control.");
  }

  await addPlotSlide(presentation, 6, {
    title: "The planner gets value from the whole belief",
    kicker: "Part 2",
    plot: "part2_planner.png",
    alt: "Planner comparison for opponent-blind, flat belief, oracle reactive, and inferred-belief planning.",
    plotFrame: { left: 56, top: 166, width: 690, height: 432 },
    textFrame: { left: 790, top: 184, width: 382, height: 220 },
    bullets: [
      "Flat-belief planner: 3.07 captures/episode.",
      "Reactive oracle with true intent: 4.05.",
      "Planner with inferred belief: 4.31.",
    ],
    callout: "The lift comes from inference plus lookahead, measured against a flat-belief ablation.",
    calloutFrame: { left: 790, top: 470, width: 394, height: 92 },
    notes: "This is the central evidence slide. The flat planner controls for generic lookahead. The inferred-belief planner improves on it, which points to opponent inference as the source of the lift.",
  });

  {
    const slide = presentation.slides.add();
    slide.background.fill = CANVAS;
    addTitle(slide, "The oracle comparison needs precise wording", "Interpretation");
    addColumnBlock(slide, {
      x: 72,
      y: 194,
      width: 306,
      title: "Oracle policy",
      color: INK,
      body: "Receives the true intent, then reacts with the trained policy. It has perfect intent information.",
    });
    addColumnBlock(slide, {
      x: 488,
      y: 194,
      width: 306,
      title: "Belief planner",
      color: ACCENT,
      body: "Receives an inferred distribution, then searches short futures with simulator dynamics.",
    });
    addColumnBlock(slide, {
      x: 904,
      y: 194,
      width: 306,
      title: "Reviewer line",
      color: WARNING,
      body: "Say it exceeds a reactive oracle. An oracle planner remains the real ceiling experiment.",
    });
    addText(slide, "This wording preserves the strongest result without overstating the comparator.", {
      left: 72,
      top: 586,
      width: 940,
      height: 48,
    }, { fontSize: 26, bold: true, color: INK });
    addFooter(slide, 7);
    addNotes(slide, "This is a defense slide. The planner has lookahead, while the oracle policy reacts. That difference explains why 4.31 can sit above 4.05 without claiming a stronger result than the experiment supports.");
  }

  await addPlotSlide(presentation, 8, {
    title: "JEPA recovers the belief without intent labels",
    kicker: "Representation",
    plot: "jepa_vs_vae_encoder.png",
    alt: "JEPA and VAE comparison for self-supervised opponent intent representation.",
    plotFrame: { left: 56, top: 166, width: 708, height: 432 },
    textFrame: { left: 808, top: 184, width: 360, height: 238 },
    bullets: [
      "Same trajectory data and the same 2-D latent size.",
      "JEPA probe accuracy: 0.89. VAE probe accuracy: 0.53.",
      "JEPA belief planner reaches 4.08 captures/episode.",
    ],
    callout: "Predicting future behavior keeps the stable intent signal.",
    calloutFrame: { left: 808, top: 496, width: 370, height: 66 },
    notes: "This slide connects the supervised belief result to a label-free route. JEPA performs better because the prediction task rewards features that explain future motion. The VAE spends capacity on reconstruction detail that does less for intent.",
  });

  await addPlotSlide(presentation, 9, {
    title: "Learned dynamics failed the control test",
    kicker: "Boundary",
    plot: "part4_jepa_world_planner.png",
    alt: "JEPA world-model planning result showing weak learned-dynamics performance.",
    plotFrame: { left: 56, top: 166, width: 690, height: 432 },
    textFrame: { left: 790, top: 184, width: 386, height: 236 },
    bullets: [
      "Simulator belief planning works.",
      "The latent world planner reaches 0.57 captures/episode.",
      "The control bottleneck is dynamics fidelity.",
    ],
    callout: "Keep this negative result in the talk. It tells reviewers where the method breaks today.",
    calloutFrame: { left: 790, top: 478, width: 386, height: 84 },
    notes: "This is the honest boundary. The encoder can discard nuisance motion, while a dynamics model must preserve details that affect control. The current data supports belief planning with simulator dynamics.",
  });

  await addPlotSlide(presentation, 10, {
    title: "BC is strong enough to test strategy inputs",
    kicker: "Follow-up",
    plot: "mopa_bc_vs_mappo.png",
    alt: "Behavior cloning capture comparison against MAPPO expert and random policy.",
    plotFrame: { left: 56, top: 166, width: 690, height: 432 },
    textFrame: { left: 790, top: 184, width: 386, height: 238 },
    bullets: [
      "Random policy: 0.40 captures/episode.",
      "Vanilla BC: 1.22 captures/episode.",
      "MAPPO expert: 1.35 captures/episode.",
    ],
    callout: "The clone recovers most of the expert edge, so the follow-up can test strategy information.",
    calloutFrame: { left: 790, top: 488, width: 386, height: 74 },
    notes: "Use this as a sanity check. A poor clone would make the latent experiment hard to read. Here the clone is close enough to MAPPO that placement information can be evaluated in deployed captures.",
  });

  {
    const slide = presentation.slides.add();
    slide.background.fill = CANVAS;
    addTitle(slide, "Strategy information has value, even when the latent is weak", "Follow-up");
    addPanel(slide, { left: 56, top: 166, width: 540, height: 346 });
    addPanel(slide, { left: 654, top: 166, width: 540, height: 346 });
    await addPlot(slide, "mopa_bc_latent_deploy.png", {
      left: 72,
      top: 182,
      width: 508,
      height: 314,
    }, "Deployment comparison for placement-blind, latent-conditioned, and oracle-placement BC.");
    await addPlot(slide, "mopa_bc_latent_sweep.png", {
      left: 670,
      top: 182,
      width: 508,
      height: 314,
    }, "Observation sweep comparing vanilla, latent-conditioned, and oracle-conditioned BC action accuracy.");
    addBullets(slide, [
      "Oracle placement closes most of the deployed capture gap.",
      "The unsupervised latent gives a small deployed lift today.",
      "The sweep shows better probes can improve BC accuracy.",
    ], { left: 82, top: 548, width: 980, height: 86 }, { fontSize: 21 });
    addFooter(slide, 11);
    addNotes(slide, "This slide gives the careful BC interpretation. True placement information matters. The current unsupervised latent has headroom. Do not sell this as the main paper win; sell it as evidence that a better strategy code could matter.");
  }

  {
    const slide = presentation.slides.add();
    slide.background.fill = CANVAS;
    addTitle(slide, "The literature slot is belief plus planning", "Positioning");
    addColumnBlock(slide, {
      x: 72,
      y: 190,
      width: 318,
      title: "Closest comparator",
      color: ACCENT,
      body: "HOP couples goal belief with tree search. It is the most direct system-level baseline to run.",
      fontSize: 21,
    });
    addColumnBlock(slide, {
      x: 474,
      y: 190,
      width: 318,
      title: "Model-based MARL",
      color: INK,
      body: "MAZero, MAMBA, MARIE, and MATWM test the planning and dynamics side of the story.",
      fontSize: 21,
    });
    addColumnBlock(slide, {
      x: 876,
      y: 190,
      width: 318,
      title: "Opponent modeling",
      color: WARNING,
      body: "MBOM and AORPO test whether the opponent model adds value beyond policy conditioning.",
      fontSize: 21,
    });
    addText(slide, "Paper line: calibrated opponent belief improves planning in a controlled hidden-intent task. Broader claims need adaptive opponents and external baselines.", {
      left: 72,
      top: 572,
      width: 1040,
      height: 62,
    }, { fontSize: 25, bold: true, color: INK });
    addFooter(slide, 12);
    addNotes(slide, "This slide arms the literature answer. HOP is the cleanest first comparison. The model-based MARL line tests world-model quality. The opponent-modeling line tests whether belief adds more than conditioning on opponent history.");
  }

  {
    const slide = presentation.slides.add();
    slide.background.fill = CANVAS;
    addTitle(slide, "Before submission, run the missing tests", "Close");
    const rows = [
      ["Oracle planner", "Measures the true planning ceiling with known intent."],
      ["Switching prey", "Tests whether the belief tracks a strategy change during an episode."],
      ["HOP baseline", "Checks the closest belief-search comparator."],
      ["Raw artifact archive", "Pins checkpoints, logs, and metrics to a release DOI."],
    ];
    let y = 186;
    for (const [label, body] of rows) {
      addRule(slide, { left: 72, top: y + 14, width: 72, height: 5 }, label === "Raw artifact archive" ? WARNING : ACCENT);
      addText(slide, label, { left: 174, top: y - 2, width: 280, height: 38 }, {
        fontSize: 25,
        bold: true,
        color: INK,
      });
      addText(slide, body, { left: 500, top: y - 2, width: 650, height: 44 }, {
        fontSize: 24,
        color: MUTED,
      });
      y += 86;
    }
    addText(slide, "End with a concrete ask: accept the controlled result, then fund the baseline and switching-opponent run.", {
      left: 72,
      top: 588,
      width: 1030,
      height: 44,
    }, { fontSize: 25, bold: true, color: INK });
    addFooter(slide, 13);
    addNotes(slide, "Close on specific work. The repo now has code hygiene, tests, result checks, and a deck. The scientific gap is the next experiment set: oracle planner, switching prey, HOP, and archived raw runs.");
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
