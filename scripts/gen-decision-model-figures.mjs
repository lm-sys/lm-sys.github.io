import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const output = fileURLToPath(
  new URL("../public/images/blog/sglang-decision-models/", import.meta.url)
);
const colors = {
  ink: "#152033",
  muted: "#49566b",
  line: "#d4dce5",
  query: "#edf3ff",
  item: "#f5f7fa",
  score: "#e6f5f1",
  Generate: "#2563eb",
  SIS: "#c66b00",
  MIS: "#008577",
};
const candidateCounts = [2, 5, 9, 16];
const series = ["Generate", "SIS", "MIS"];
const candidateData = [
  {
    model: "Qwen3-0.6B",
    Generate: [18.9, 24.5, 30.8, 39.6],
    SIS: [17.0, 19.9, 20.9, 24.3],
    MIS: [17.5, 17.8, 17.7, 18.7],
  },
  {
    model: "Qwen3-8B",
    Generate: [21.7, 32.7, 45.8, 54.1],
    SIS: [22.2, 32.2, 43.0, 53.1],
    MIS: [20.2, 24.3, 20.6, 20.6],
  },
  {
    model: "Qwen3.5-4B",
    Generate: [57.8, 64.1, 68.3, 84.8],
    SIS: [53.7, 58.4, 60.0, 74.4],
    MIS: [55.7, 55.6, 55.5, 55.7],
  },
];

function escape(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;",
  }[character]));
}

function text(left, top, value, size = 24, attributes = "") {
  const lines = Array.isArray(value) ? value : [value];
  return `<text x="${left}" y="${top}" font-size="${size}" ${attributes}>${lines
    .map((line, index) => `<tspan x="${left}" dy="${index ? size * 1.4 : 0}">${escape(line)}</tspan>`)
    .join("")}</text>`;
}

function box(left, top, width, height, fill) {
  return `<rect x="${left}" y="${top}" width="${width}" height="${height}" rx="6" fill="${fill}" stroke="${colors.line}"/>`;
}

function arrow(start, top, end) {
  return `<path d="M ${start} ${top} H ${end}" fill="none" stroke="${colors.muted}" stroke-width="2" marker-end="url(#arrow)"/>`;
}

function rule(top) {
  return `<path d="M 40 ${top} H 920" stroke="${colors.line}"/>`;
}

function svg(title, description, height, content, provenance = "") {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="960" height="${height}" viewBox="0 0 960 ${height}" role="img" aria-labelledby="title description">
<title id="title">${escape(title)}</title>
<desc id="description">${escape(description)}</desc>
<metadata>${escape(provenance)}</metadata>
<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M 0 0 L 8 4 L 0 8 Z" fill="${colors.muted}"/></marker></defs>
<rect width="960" height="${height}" fill="white"/>
<g fill="${colors.ink}" font-family="Avenir Next, DejaVu Sans, sans-serif">${content}</g>
</svg>\n`;
}

function promptFigure() {
  const parts = [
    text(40, 52, "What can each judgment see?", 34, 'font-weight="700"'),
    text(40, 94, "The same actions, two prompt constructions", 24, `fill="${colors.muted}"`),
    text(40, 159, "Pointwise", 29, 'font-weight="700"'),
    text(40, 197, "One independent Yes/No judgment per candidate", 24),
  ];
  ["A: Order-status service", "B: Delivery-policy search", "C: Ask for the order ID"].forEach((label, index) => {
    const top = 224 + index * 86;
    parts.push(
      box(40, top, 256, 64, colors.query),
      text(58, top + 40, "Context + question", 24),
      arrow(300, top + 32, 323),
      box(330, top, 320, 64, colors.item),
      text(346, top + 40, label, 23),
      arrow(654, top + 32, 690),
      box(700, top, 220, 64, colors.score),
      text(810, top + 40, "[Yes, No]", 25, 'text-anchor="middle"'),
    );
  });
  parts.push(
    text(40, 501, "Each prompt ends with the same Yes/No instruction.", 23, `fill="${colors.muted}"`),
    rule(534),
    text(40, 586, "Setwise", 29, 'font-weight="700"'),
    text(40, 624, "One A/B/C judgment after seeing all candidates", 24),
    box(40, 654, 610, 156, colors.query),
    text(60, 693, ["Context + question", "A: Order status  |  B: Delivery-policy search", "C: Ask for the order ID", "Choose A, B, or C."], 22),
    arrow(654, 732, 690),
    box(700, 700, 220, 64, colors.score),
    text(810, 740, "[A, B, C]", 25, 'text-anchor="middle"'),
    text(40, 861, "Prompt semantics are separate from SIS or MIS execution.", 23, `fill="${colors.muted}"`),
  );
  return svg(
    "Pointwise and setwise prompt construction",
    "Pointwise: three separate context-plus-candidate prompts yield three Yes/No rows. Setwise: one prompt containing all three candidates yields one A/B/C row.",
    900,
    parts.join("\n"),
  );
}

function contractFigure() {
  const parts = [
    text(40, 52, "Two independent serving improvements", 34, 'font-weight="700"'),
    text(40, 113, "1. Make the output contract explicit", 29, 'font-weight="700"'),
    text(40, 158, "One-token generation", 25, 'font-weight="600"'),
    text(510, 158, "Score API: /v1/score", 25, 'font-weight="600"'),
    box(40, 180, 410, 110, colors.item),
    text(60, 218, ["Top-k logprobs", "A required label may be absent."], 24),
    box(510, 180, 410, 110, colors.score),
    text(530, 218, ["Explicit label scores", "Requested labels, in order."], 24),
    rule(324),
    text(40, 378, "2. Reuse the query across candidates", 29, 'font-weight="700"'),
    text(40, 423, "SIS: separate logical sequences", 25, 'font-weight="600"'),
  ];
  ["A", "B", "C"].forEach((label, index) => {
    const top = 447 + index * 67;
    parts.push(
      box(40, top, 260, 50, colors.query),
      text(170, top + 33, "Query", 24, 'text-anchor="middle"'),
      arrow(304, top + 25, 331),
      box(340, top, 260, 50, colors.item),
      text(470, top + 33, `Candidate ${label}`, 24, 'text-anchor="middle"'),
      arrow(604, top + 25, 650),
      box(660, top, 260, 50, colors.score),
      text(790, top + 33, "[Yes, No]", 24, 'text-anchor="middle"'),
    );
  });
  parts.push(
    text(40, 671, "One request can contain all three sequences.", 23, `fill="${colors.muted}"`),
    rule(704),
    text(40, 753, "MIS: shared query, isolated candidates", 25, 'font-weight="600"'),
    box(40, 780, 260, 184, colors.query),
    text(170, 857, ["Query", "computed once"], 24, 'text-anchor="middle"'),
    `<path d="M 300 872 H 320 M 320 805 V 939" fill="none" stroke="${colors.muted}" stroke-width="2"/>`,
  );
  ["A", "B", "C"].forEach((label, index) => {
    const top = 780 + index * 67;
    parts.push(
      arrow(320, top + 25, 331),
      box(340, top, 260, 50, colors.item),
      text(470, top + 33, `Candidate ${label}`, 24, 'text-anchor="middle"'),
      arrow(604, top + 25, 650),
      box(660, top, 260, 50, colors.score),
      text(790, top + 33, "[Yes, No]", 24, 'text-anchor="middle"'),
    );
  });
  parts.push(
    text(40, 1004, "Each candidate attends only to the query and its own tokens.", 23, `fill="${colors.muted}"`),
    rule(1038),
    text(40, 1084, "Both scoring and one-token generation can finish from prefill.", 23),
    text(40, 1124, ["Label extraction still requires the vocabulary projection", "and full-distribution normalization."], 23, `fill="${colors.muted}"`),
  );
  return svg(
    "Explicit scoring and shared-query execution",
    "The Score API returns requested labels explicitly. Independently, MIS shares query computation while isolating candidates; SIS uses separate logical sequences. Neither benefit implies an extra generation forward pass.",
    1200,
    parts.join("\n"),
  );
}

function candidateFigure() {
  const parts = [
    text(40, 49, "P95 time to decision by candidate count", 33, 'font-weight="700"'),
    text(40, 88, "Latency in milliseconds; the same scale in every panel", 23, `fill="${colors.muted}"`),
  ];
  series.forEach((name, index) => {
    const left = 40 + index * 204;
    parts.push(
      `<rect x="${left}" y="113" width="28" height="18" fill="${colors[name]}"/>`,
      text(left + 40, 132, name, 24),
    );
  });
  candidateData.forEach((model, modelIndex) => {
    const panelTop = 183 + modelIndex * 420;
    const plotTop = panelTop + 30;
    const baseline = plotTop + 300;
    parts.push(text(40, panelTop, model.model, 29, 'font-weight="700"'));
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100].forEach((tick) => {
      const top = baseline - tick * 3;
      parts.push(
        `<path d="M 92 ${top} H 920" stroke="${colors.line}"/>`,
        text(76, top + 7, tick, 21, `text-anchor="end" fill="${colors.muted}"`),
      );
    });
    candidateCounts.forEach((count, countIndex) => {
      const center = 194 + countIndex * 207;
      series.forEach((name, seriesIndex) => {
        assert.equal(model[name].length, candidateCounts.length);
        const value = model[name][countIndex];
        assert(Number.isFinite(value) && value >= 0 && value <= 100);
        const left = center - 78 + seriesIndex * 54;
        const height = value * 3;
        parts.push(
          `<rect x="${left}" y="${baseline - height}" width="48" height="${height}" fill="${colors[name]}"><title>${escape(`${model.model}, ${count} candidates, ${name}: ${value.toFixed(1)} ms`)}</title></rect>`,
          text(left + 24, baseline - height - 10, value.toFixed(1), 21, 'text-anchor="middle"'),
        );
      });
      parts.push(text(center, baseline + 33, count, 24, 'text-anchor="middle"'));
    });
  });
  parts.push(text(506, 1440, "Number of candidates", 25, 'text-anchor="middle"'));
  return svg(
    "P95 time to decision by candidate count",
    "Grouped bars for 2, 5, 9, and 16 candidates on Qwen3-0.6B, Qwen3-8B, and Qwen3.5-4B. Values are displayed to one decimal place. All panels use a zero baseline and a 0 to 100 ms scale with 10 ms ticks.",
    1470,
    parts.join("\n"),
    "Values transcribed from the original figure's one-decimal labels, with a user-provided correction to Qwen3-8B SIS at 16 candidates: 53.1 ms. Raw benchmark samples were not supplied. The load chart is not regenerated because its source measurements were not supplied.",
  );
}

await mkdir(output, { recursive: true });
for (const [name, source] of [
  ["pointwise-vs-setwise", promptFigure()],
  ["scoring-contract", contractFigure()],
  ["pointwise-latency-by-candidates", candidateFigure()],
]) {
  await writeFile(`${output}${name}.svg`, source);
  console.log(`${name}: vector SVG`);
}