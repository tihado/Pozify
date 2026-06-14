import React, {
  useEffect,
  useMemo,
  useRef,
  useState,
} from "https://esm.sh/react@18.2.0";
import { createRoot } from "https://esm.sh/react-dom@18.2.0/client";

const h = React.createElement;

const defaults = {
  description:
    "Upload a short workout clip, tune the athlete context, and generate an annotated form-review report with structured artifacts.",
  goals: [
    "strength",
    "hypertrophy",
    "endurance",
    "mobility",
    "beginner_practice",
  ],
  experience_levels: ["beginner", "intermediate"],
  exercises: ["auto"],
  limitations: ["wrist_discomfort", "knee_discomfort", "shoulder_discomfort"],
  equipment: ["bodyweight", "dumbbell", "barbell", "unknown"],
};

function label(value) {
  return value.replaceAll("_", " ");
}

function Field({ labelText, children, full = false }) {
  return h(
    "label",
    { className: `field${full ? " full" : ""}` },
    h("span", { className: "label" }, labelText),
    children,
  );
}

function SelectField({ labelText, value, onChange, options, name }) {
  return h(
    Field,
    { labelText },
    h(
      "select",
      { name, value, onChange: (event) => onChange(event.target.value) },
      options.map((option) =>
        h("option", { key: option, value: option }, label(option)),
      ),
    ),
  );
}

function formatValue(value) {
  if (typeof value === "number")
    return Number.isInteger(value) ? String(value) : value.toFixed(3);
  if (Array.isArray(value)) return value.map(formatValue).join(", ");
  if (value && typeof value === "object") return JSON.stringify(value);
  return value ?? "n/a";
}

function percent(value) {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : "n/a";
}

function KineticFigure({ compact = false }) {
  const imageUrl = compact
    ? "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?auto=format&fit=crop&fm=jpg&w=900&q=84"
    : "https://images.unsplash.com/photo-1517836357463-d25dfeac3438?auto=format&fit=crop&fm=jpg&w=1100&q=84";
  return h(
    "div",
    { className: `kinetic-photo${compact ? " compact" : ""}`, "aria-hidden": "true" },
    h("img", {
      alt: "",
      className: "motion-photo",
      draggable: false,
      src: imageUrl,
    }),
    h("span", { className: "photo-vignette" }),
    h("span", { className: "photo-scanline" }),
  );
}

function SignalStack() {
  return h(
    "div",
    { className: "signal-stack", "aria-label": "Analysis pipeline" },
    ["Video QC", "Pose Map", "Rep Count", "Coach Notes"].map((item, index) =>
      h(
        "div",
        { className: "signal-row", key: item },
        h("span", null, `0${index + 1}`),
        h("strong", null, item),
      ),
    ),
  );
}

const runningProgressSteps = [
  {
    id: "quality",
    text: "First up, I am checking if the video is clear enough to coach from.",
    delayMs: 0,
  },
  {
    id: "pose",
    text: "Now I am mapping your posture and tracking the key body landmarks.",
    delayMs: 900,
  },
  {
    id: "exercise",
    text: "Let me figure out which exercise you are doing.",
    delayMs: 1800,
  },
  {
    id: "reps",
    text: "Counting your reps now. One clean rep at a time.",
    delayMs: 2800,
  },
  {
    id: "issues",
    text: "Almost there. I am checking the moments that may need a small fix.",
    delayMs: 3900,
  },
  {
    id: "render",
    text: "I am preparing your annotated video and issue clips.",
    delayMs: 4800,
  },
  {
    id: "coach",
    text: "I am turning the scan into coaching notes you can use right away.",
    delayMs: 5600,
  },
];

function pendingProgressState() {
  return runningProgressSteps.map((step, index) => ({
    id: step.id,
    text: step.text,
    status: index === 0 ? "active" : "pending",
  }));
}

function finalProgressState(result) {
  const report = result.report;
  const warnings = report.video_manifest?.quality_warnings || [];
  const exercise = label(report.exercise?.exercise || "movement");
  const repCount = report.reps?.reps?.length || 0;
  const issues = report.issue_markers?.issues || [];
  return [
    {
      id: "quality",
      status: "done",
      text: warnings.length
        ? `Quick note: the video has a few things to watch, like ${warnings.map(label).join(", ")}.`
        : "Nice, your video quality looks solid.",
    },
    {
      id: "pose",
      status: "done",
      text: "Posture tracking is done. I found the key landmarks I need.",
    },
    {
      id: "exercise",
      status: "done",
      text: `Looks like you are doing ${exercise}.`,
    },
    {
      id: "reps",
      status: "done",
      text: `I counted ${repCount} ${exercise} reps in this set.`,
    },
    {
      id: "issues",
      status: "done",
      text: issues.length
        ? `I found ${issues.length} coaching point${issues.length === 1 ? "" : "s"} worth reviewing.`
        : "Good news, I did not spot any clear form issues in this set.",
    },
    {
      id: "render",
      status: "done",
      text: result.annotated_video_url
        ? "Your annotated video is ready."
        : "I could not render an annotated video, but the report is ready.",
    },
    {
      id: "coach",
      status: "done",
      text: "Coach notes are ready.",
    },
  ];
}

function applyProgressEvent(currentSteps, event) {
  const baseSteps = currentSteps.length ? currentSteps : pendingProgressState();
  const knownStepIds = new Set(runningProgressSteps.map((step) => step.id));
  if (!knownStepIds.has(event.step)) return baseSteps;
  return baseSteps.map((step) =>
    step.id === event.step
      ? {
          ...step,
          status: event.status || "active",
          text: event.text || step.text,
        }
      : step,
  );
}

async function readAnalysisStream(response, onEvent) {
  if (!response.body)
    throw new Error("Streaming is not available in this browser.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line);
      if (event.type === "progress") onEvent(event);
      if (event.type === "complete") result = event.result;
      if (event.type === "error")
        throw new Error(event.detail || "Analysis failed.");
    }
  }

  if (buffer.trim()) {
    const event = JSON.parse(buffer);
    if (event.type === "progress") onEvent(event);
    if (event.type === "complete") result = event.result;
    if (event.type === "error")
      throw new Error(event.detail || "Analysis failed.");
  }

  if (!result) throw new Error("Analysis finished without a report.");
  return result;
}

function ProgressPanel({ steps }) {
  if (!steps.length) return null;
  const isComplete = steps.every((step) => step.status === "done");
  return h(
    "section",
    { className: "progress-panel", "aria-live": "polite" },
    h(
      "h3",
      null,
      isComplete ? "Scan results are ready" : "Your scan is moving",
    ),
    h(
      "ol",
      { className: "progress-list" },
      steps.map((step) =>
        h(
          "li",
          { className: `progress-step ${step.status}`, key: step.id },
          h("span", { className: "progress-dot", "aria-hidden": "true" }),
          h("span", null, step.text),
        ),
      ),
    ),
  );
}

function StageEmpty() {
  return h(
    "div",
    { className: "result-empty" },
    h(KineticFigure, { compact: true }),
    h("strong", null, "Your annotated replay will land here"),
    h(
      "span",
      null,
      "Upload a set and Pozify will paint the movement path, rep timing, and coaching moments on top of the video.",
    ),
  );
}

function ReviewInsights({ result }) {
  if (!result) return null;
  const report = result.report;
  const warnings = report.video_manifest?.quality_warnings || [];
  const exercise = label(report.exercise?.exercise || "movement");
  const repCount = report.reps?.reps?.length || 0;
  const issues = report.issue_markers?.issues || [];
  const confidence = percent(report.exercise?.confidence);
  const issueText = issues.length
    ? `${issues.length} coaching moment${issues.length === 1 ? "" : "s"}`
    : "No clear form issues";

  return h(
    "div",
    { className: "scan-insights", "aria-label": "Scan results" },
    h(
      "article",
      { className: "scan-insight" },
      h("span", null, "Movement"),
      h("strong", null, exercise),
      h("small", null, `confidence ${confidence}`),
    ),
    h(
      "article",
      { className: "scan-insight" },
      h("span", null, "Reps counted"),
      h("strong", null, String(repCount)),
      h("small", null, repCount === 1 ? "clean rep detected" : "reps detected"),
    ),
    h(
      "article",
      { className: "scan-insight" },
      h("span", null, "Coach review"),
      h("strong", null, issueText),
      h(
        "small",
        null,
        issues.length ? "tap Issues for clips" : "nothing major stood out",
      ),
    ),
    h(
      "article",
      { className: "scan-insight" },
      h("span", null, "Video quality"),
      h("strong", null, warnings.length ? "Needs a quick note" : "Looks good"),
      h(
        "small",
        null,
        warnings.length
          ? warnings.map(label).join(", ")
          : "clear enough to coach",
      ),
    ),
  );
}

function metricScore(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.max(0, Math.min(1, value))
    : null;
}

function metricScoreRows(report) {
  const aggregate = report.rep_analysis?.aggregate_metrics || {};
  return [
    ["ROM", aggregate.avg_rom_score],
    ["Stability", aggregate.avg_stability_score],
    ["Symmetry", aggregate.avg_symmetry_score],
    ["Pose coverage", aggregate.pose_valid_ratio],
  ]
    .map(([name, value]) => ({ name, value: metricScore(value) }))
    .filter((item) => item.value !== null);
}

function scoreTone(score) {
  if (score === null) return "neutral";
  if (score >= 0.8) return "strong";
  if (score >= 0.62) return "watch";
  return "risk";
}

function repScore(item) {
  if (!item) return null;
  const values = [
    item.range_of_motion_score,
    item.stability_score,
    item.symmetry_score,
  ]
    .map(metricScore)
    .filter((value) => value !== null);
  if (!values.length) return null;
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function repAnalysisById(report) {
  return new Map(
    (report.rep_analysis?.items || []).map((item) => [item.rep_id, item]),
  );
}

function issueList(report) {
  return report.issue_markers?.issues || [];
}

function issuesForRep(report, repId) {
  return issueList(report).filter((issue) => issue.rep_id === repId);
}

function topIssue(report) {
  return [...issueList(report)].sort((a, b) => b.severity - a.severity)[0] || null;
}

function bestRep(report) {
  const analysisById = repAnalysisById(report);
  return (report.reps?.reps || [])
    .map((rep) => {
      const score = repScore(analysisById.get(rep.rep_id));
      return { rep, score };
    })
    .filter((entry) => entry.score !== null)
    .sort((a, b) => b.score - a.score)[0] || null;
}

function reportDuration(report) {
  const manifestDuration = report.video_manifest?.duration_sec;
  const repEnd = Math.max(
    0,
    ...(report.reps?.reps || []).map((rep) => rep.end_sec || 0),
    ...issueList(report).map((issue) => issue.end_sec || 0),
  );
  return Math.max(manifestDuration || 0, repEnd, 1);
}

function repAtTime(report, playbackTime) {
  return (
    (report.reps?.reps || []).find(
      (rep) => playbackTime >= rep.start_sec && playbackTime <= rep.end_sec,
    ) || null
  );
}

function qualityVerdict(report) {
  const scores = metricScoreRows(report)
    .filter((row) => row.name !== "Pose coverage")
    .map((row) => row.value);
  const average = scores.length
    ? scores.reduce((total, score) => total + score, 0) / scores.length
    : null;
  const issue = topIssue(report);
  if (average === null) {
    return {
      title: "Report ready",
      detail: "Coach notes are grounded in the scan.",
      score: null,
    };
  }
  const adjusted = Math.max(0, average - (issue?.severity || 0) * 0.08);
  if (adjusted >= 0.82) {
    return {
      title: issue ? "Strong set, one watchpoint" : "Clean training set",
      detail: issue ? `${issueTitle(issue)} is the main limiter.` : "No clear form issue dominated the set.",
      score: adjusted,
    };
  }
  if (adjusted >= 0.66) {
    return {
      title: "Usable reps, focused fix",
      detail: issue ? `${issueTitle(issue)} should be fixed first.` : "Quality is workable, but keep reps controlled.",
      score: adjusted,
    };
  }
  return {
    title: "Needs cleaner reps",
    detail: issue ? `${issueTitle(issue)} is limiting this set.` : "Repeat with slower reps before progressing.",
    score: adjusted,
  };
}

function timeRange(start, end) {
  return `${Number(start || 0).toFixed(2)}s-${Number(end || 0).toFixed(2)}s`;
}

function metaChip(text, tone = "", key = undefined) {
  const props = { className: `meta-chip${tone ? ` ${tone}` : ""}` };
  if (key !== undefined) props.key = key;
  return h("span", props, text);
}

function ScoreBar({ labelText, value, detail }) {
  const score = metricScore(value);
  const width = score === null ? "0%" : `${Math.round(score * 100)}%`;
  return h(
    "div",
    { className: `score-card ${scoreTone(score)}` },
    h(
      "div",
      { className: "score-card-head" },
      h("span", null, labelText),
      h("strong", null, score === null ? "n/a" : percent(score)),
    ),
    h(
      "span",
      { className: "score-track", "aria-hidden": "true" },
      h("span", { className: "score-fill", style: { width } }),
    ),
    detail ? h("small", null, detail) : null,
  );
}

function BriefTile({ eyebrow, title, body, tone = "" }) {
  return h(
    "article",
    { className: `brief-tile${tone ? ` ${tone}` : ""}` },
    h("span", null, eyebrow),
    h("strong", null, title),
    h("p", null, body),
  );
}

function RepTimeline({
  report,
  compact = false,
  activeRepId = null,
  playbackTime = 0,
  onRepSelect = null,
}) {
  const reps = report.reps?.reps || [];
  const duration = reportDuration(report);
  const analysisById = repAnalysisById(report);
  const isPlayback = typeof onRepSelect === "function";
  const playheadLeft = `${Math.max(0, Math.min(100, (playbackTime / duration) * 100))}%`;
  if (!reps.length) {
    return h("p", { className: "empty-copy" }, "No complete reps were detected.");
  }

  return h(
    "div",
    {
      className: `rep-timeline${compact ? " compact" : ""}${isPlayback ? " playback" : ""}`,
    },
    h(
      "div",
      { className: "timeline-ruler", "aria-hidden": "true" },
      h("span", null, "0s"),
      h("span", null, `${duration.toFixed(1)}s`),
    ),
    h(
      "div",
      {
        className: `rep-track${isPlayback ? " interactive" : ""}`,
        "aria-label": isPlayback ? "Interactive rep timeline" : "Rep timeline",
      },
      isPlayback
        ? h("span", {
            className: "timeline-playhead",
            style: { left: playheadLeft },
            "aria-hidden": "true",
          })
        : null,
      reps.map((rep) => {
        const left = `${Math.max(0, Math.min(98, (rep.start_sec / duration) * 100))}%`;
        const width = `${Math.max(6, Math.min(100, ((rep.end_sec - rep.start_sec) / duration) * 100))}%`;
        const score = repScore(analysisById.get(rep.rep_id));
        const repIssues = issuesForRep(report, rep.rep_id);
        const tagName = isPlayback ? "button" : "div";
        const segmentProps = {
          className: `rep-segment ${scoreTone(score)}${repIssues.length ? " has-issue" : ""}${activeRepId === rep.rep_id ? " active" : ""}`,
          key: rep.rep_id,
          style: { left, width },
          title: `Rep ${rep.rep_id} ${timeRange(rep.start_sec, rep.end_sec)}`,
        };
        if (isPlayback) {
          segmentProps.type = "button";
          segmentProps.onClick = () => onRepSelect(rep);
          segmentProps["aria-label"] = `Play from rep ${rep.rep_id}, ${timeRange(rep.start_sec, rep.end_sec)}`;
          segmentProps["aria-pressed"] = activeRepId === rep.rep_id;
        }
        return h(
          tagName,
          segmentProps,
          h("strong", null, `R${rep.rep_id}`),
          repIssues.map((issue, index) =>
            h("span", {
              className: `issue-pin ${severityLevel(issue)}`,
              key: `${issue.issue}-${index}`,
              style: {
                left: `${Math.max(8, Math.min(92, ((issue.start_sec - rep.start_sec) / Math.max(0.01, rep.end_sec - rep.start_sec)) * 100))}%`,
              },
              title: issueTitle(issue),
            }),
          ),
        );
      }),
    ),
  );
}

function SummaryTab({ result }) {
  if (!result) {
    return h(
      "section",
      { className: "summary" },
      h("h2", null, "Ready for review"),
      h("p", null, "Choose the movement context and start the analyzer."),
    );
  }

  const report = result.report;
  const summary = report.coach_summary;
  const warnings = report.video_manifest.quality_warnings || [];
  const issues = issueList(report);
  const issue = topIssue(report);
  const best = bestRep(report);
  const verdict = qualityVerdict(report);
  const scores = metricScoreRows(report);
  const provider = report.artifacts?.coach_summary_provider || "n/a";
  const model = report.artifacts?.coach_summary_model || "n/a";
  const repCount = report.reps?.reps?.length || 0;
  const cueNow =
    summary.top_fixes?.[0] ||
    (issue
      ? issueEvidence(issue)
      : "Repeat the next set with the same controlled tempo.");
  const nextSession = summary.next_session_plan?.[0] || "Repeat the set with controlled reps.";

  return h(
    "section",
    { className: "summary overview-panel" },
    h(
      "div",
      { className: "report-hero" },
      h(
        "div",
        { className: "report-copy" },
        h("p", { className: "eyebrow" }, "Movement report"),
        h("h2", null, `${label(report.exercise.exercise)} review`),
        h("p", { className: "report-lede" }, summary.summary),
        h(
          "div",
          { className: "pill-row" },
          metaChip(`confidence ${percent(report.exercise.confidence)}`, "blue"),
          metaChip(`${repCount} rep${repCount === 1 ? "" : "s"}`),
          metaChip(
            issues.length
              ? `${issues.length} coaching moment${issues.length === 1 ? "" : "s"}`
              : "no clear issue",
            issues.length ? severityLevel(issue) : "strong",
          ),
          metaChip(provider === "local_transformers" ? "local coach" : provider),
        ),
      ),
      h(
        "aside",
        { className: `verdict-card ${scoreTone(verdict.score)}` },
        h("span", null, "Set verdict"),
        h("strong", null, verdict.title),
        h("p", null, verdict.detail),
        h("small", null, model),
      ),
    ),
    h(
      "div",
      { className: "brief-grid" },
      h(BriefTile, {
        eyebrow: "Best rep",
        title: best ? `Rep ${best.rep.rep_id}` : "Not enough data",
        body: best
          ? `Most consistent rep at ${timeRange(best.rep.start_sec, best.rep.end_sec)} with ${percent(best.score)} quality.`
          : "Complete reps are needed before the app can pick a best rep.",
        tone: "mint",
      }),
      h(BriefTile, {
        eyebrow: "Main limiter",
        title: issue ? issueTitle(issue) : "Nothing major",
        body: issue
          ? `${issueClipText(result, issue, 0)}. ${issueEvidence(issue)}.`
          : "No sustained threshold violations were found in this set.",
        tone: issue ? severityLevel(issue) : "mint",
      }),
      h(BriefTile, {
        eyebrow: "Cue now",
        title: "Try this first",
        body: cueNow,
        tone: "volt",
      }),
      h(BriefTile, {
        eyebrow: "Next session",
        title: "Keep it simple",
        body: nextSession,
      }),
    ),
    h(
      "div",
      { className: "score-board" },
      scores.length
        ? scores.map((row) =>
            h(ScoreBar, {
              key: row.name,
              labelText: row.name,
              value: row.value,
            }),
          )
        : h("p", { className: "empty-copy" }, "No movement scores available."),
    ),
    h(RepTimeline, { report, compact: true }),
    h(
      "div",
      { className: "note-grid report-notes" },
      h(NoteList, { title: "What you did", items: summary.what_you_did }),
      h(NoteList, { title: "What looked good", items: summary.what_looked_good }),
      h(NoteList, { title: "What changed", items: summary.what_changed_across_reps }),
    ),
    warnings.length
      ? h(
          "div",
          { className: "quality-list" },
          warnings.map((warning) =>
            h("span", { key: warning }, label(warning)),
          ),
        )
      : h(
          "div",
          { className: "quality-list" },
          h("span", null, "No quality warnings"),
        ),
  );
}

function MetricsTab({ result }) {
  if (!result)
    return h(
      "section",
      { className: "summary" },
      h("h2", null, "Movement metrics"),
      h("p", null, "Metrics appear after analysis."),
    );
  const report = result.report;
  const aggregate = Object.entries(
    report.rep_analysis?.aggregate_metrics || {},
  );
  const reps = report.rep_analysis?.items || [];
  const scoreRows = metricScoreRows(report);
  return h(
    "section",
    { className: "summary metrics-panel" },
    h("h2", null, "Movement metrics"),
    h("p", null, "Score bars show the movement qualities that most directly shape the coach note."),
    h(
      "div",
      { className: "score-board metric-score-board" },
      scoreRows.length
        ? scoreRows.map((row) =>
            h(ScoreBar, {
              key: row.name,
              labelText: row.name,
              value: row.value,
            }),
          )
        : h("p", { className: "empty-copy" }, "No aggregate scores available."),
    ),
    h(
      "div",
      { className: "rep-score-list" },
      reps.length
        ? reps.map((rep) =>
            h(
              "article",
              { className: `rep-score-row ${scoreTone(repScore(rep))}`, key: rep.rep_id },
              h(
                "div",
                { className: "rep-score-title" },
                h("strong", null, `Rep ${rep.rep_id}`),
                h("span", null, `${formatValue(rep.duration_sec)}s`),
              ),
              h(ScoreBar, {
                labelText: "ROM",
                value: rep.range_of_motion_score,
              }),
              h(ScoreBar, {
                labelText: "Stability",
                value: rep.stability_score,
              }),
              h(ScoreBar, {
                labelText: "Symmetry",
                value: rep.symmetry_score,
              }),
            ),
          )
        : h("p", { className: "empty-copy" }, "No per-rep metrics available."),
    ),
    h(
      "div",
      { className: "table-wrap" },
      h(
        "table",
        null,
        h(
          "thead",
          null,
          h(
            "tr",
            null,
            ["Rep", "Duration", "ROM", "Stability", "Symmetry"].map((heading) =>
              h("th", { key: heading }, heading),
            ),
          ),
        ),
        h(
          "tbody",
          null,
          reps.map((rep) =>
            h(
              "tr",
              { key: rep.rep_id },
              h("td", null, rep.rep_id),
              h("td", null, `${formatValue(rep.duration_sec)}s`),
              h("td", null, percent(rep.range_of_motion_score)),
              h("td", null, percent(rep.stability_score)),
              h("td", null, percent(rep.symmetry_score)),
            ),
          ),
        ),
      ),
    ),
  );
}

function RepDetailPanel({ report, rep, result, playbackTime }) {
  if (!rep) {
    return h(
      "aside",
      { className: "rep-detail-panel" },
      h("span", { className: "rep-detail-kicker" }, "Current rep"),
      h("strong", null, "No rep selected"),
      h("p", null, "Play the video or click a rep segment to inspect its metrics."),
    );
  }

  const analysis = repAnalysisById(report).get(rep.rep_id);
  const repIssues = issuesForRep(report, rep.rep_id);
  const score = repScore(analysis);
  const duration = Math.max(0, (rep.end_sec || 0) - (rep.start_sec || 0));
  const firstHalfDuration = Math.max(0, (rep.mid_sec || 0) - (rep.start_sec || 0));
  const secondHalfDuration = Math.max(0, (rep.end_sec || 0) - (rep.mid_sec || 0));
  const metricRows = [
    { name: "ROM", value: metricScore(analysis?.range_of_motion_score) },
    { name: "Stability", value: metricScore(analysis?.stability_score) },
    { name: "Symmetry", value: metricScore(analysis?.symmetry_score) },
  ].filter((item) => item.value !== null);
  const weakestMetric = metricRows.length
    ? [...metricRows].sort((a, b) => a.value - b.value)[0]
    : null;
  const repProgress =
    duration > 0
      ? Math.max(0, Math.min(1, (playbackTime - rep.start_sec) / duration))
      : 0;
  const phase =
    playbackTime < rep.start_sec
      ? "setup"
      : playbackTime > rep.end_sec
        ? "review"
        : playbackTime <= rep.mid_sec
          ? "half 1"
          : "half 2";
  const focusText = weakestMetric
    ? `${weakestMetric.name} is the main review cue at ${percent(weakestMetric.value)}.`
    : "Metric detail is not available for this rep.";
  const scoreText =
    score === null
      ? "Metric data is limited for this rep."
      : score >= 0.8
        ? "This rep is one of the steadier parts of the set."
        : score >= 0.62
          ? "This rep is usable, but one metric deserves attention."
          : "This rep is the kind to slow down and review before progressing.";
  const issueText = repIssues.length
    ? `${repIssues.length} issue marker${repIssues.length === 1 ? "" : "s"} attached to this rep.`
    : "No issue marker is attached, so use the lowest metric as the review cue.";

  return h(
    "aside",
    { className: `rep-detail-panel ${scoreTone(score)}` },
    h(
      "div",
      { className: "rep-detail-head" },
      h(
        "div",
        null,
        h("span", { className: "rep-detail-kicker" }, "Current rep"),
        h("strong", null, `Rep ${rep.rep_id}`),
        h("small", null, timeRange(rep.start_sec, rep.end_sec)),
      ),
      metaChip(score === null ? "no score" : percent(score), scoreTone(score)),
    ),
    h(
      "div",
      { className: "rep-fact-grid" },
      h(
        "div",
        { className: "rep-fact" },
        h("span", null, "Duration"),
        h("strong", null, `${duration.toFixed(2)}s`),
      ),
      h(
        "div",
        { className: "rep-fact" },
        h("span", null, "Frames"),
        h("strong", null, `${rep.start_frame}-${rep.end_frame}`),
      ),
      h(
        "div",
        { className: "rep-fact" },
        h("span", null, "Tempo split"),
        h("strong", null, `${firstHalfDuration.toFixed(2)}/${secondHalfDuration.toFixed(2)}s`),
      ),
      h(
        "div",
        { className: "rep-fact" },
        h("span", null, "Phase"),
        h("strong", null, phase),
      ),
      h(
        "div",
        { className: "rep-fact" },
        h("span", null, "Focus"),
        h("strong", null, weakestMetric ? `${weakestMetric.name} ${percent(weakestMetric.value)}` : "n/a"),
      ),
      h(
        "div",
        { className: "rep-fact" },
        h("span", null, "Issues"),
        h("strong", null, repIssues.length ? String(repIssues.length) : "clear"),
      ),
    ),
    h(
      "div",
      { className: "rep-progress-card" },
      h(
        "div",
        { className: "rep-now" },
        h("span", null, "playhead"),
        h("strong", null, `${playbackTime.toFixed(2)}s`),
      ),
      h(
        "div",
        { className: "rep-progress-track", "aria-hidden": "true" },
        h("span", {
          className: "rep-progress-fill",
          style: { width: `${Math.round(repProgress * 100)}%` },
        }),
        h("span", { className: "rep-progress-mid" }),
      ),
      h(
        "div",
        { className: "rep-progress-labels" },
        h("span", null, `${rep.start_sec.toFixed(2)}s`),
        h("span", null, `${rep.mid_sec.toFixed(2)}s mid`),
        h("span", null, `${rep.end_sec.toFixed(2)}s`),
      ),
    ),
    h(
      "article",
      { className: "rep-readout" },
      h("span", null, "Movement read"),
      h("strong", null, scoreText),
      h("p", null, `${focusText} ${issueText}`),
    ),
    h(
      "div",
      { className: "rep-mini-metrics" },
      h(ScoreBar, {
        labelText: "ROM",
        value: analysis?.range_of_motion_score,
      }),
      h(ScoreBar, {
        labelText: "Stability",
        value: analysis?.stability_score,
      }),
      h(ScoreBar, {
        labelText: "Symmetry",
        value: analysis?.symmetry_score,
      }),
    ),
    h(
      "div",
      { className: "issue-mini-list" },
      h("h3", null, repIssues.length ? "Issue in this rep" : "Issue check"),
      repIssues.length
        ? repIssues.map((issue, index) =>
            h(
              "article",
              { className: `issue-mini ${severityLevel(issue)}`, key: `${issue.issue}-${index}` },
              h(
                "div",
                { className: "issue-mini-head" },
                h("strong", null, issueTitle(issue)),
                h(
                  "span",
                  { className: `severity-chip ${severityLevel(issue)}` },
                  severityText(issue),
                ),
              ),
              h("p", null, issueEvidence(issue)),
              h(
                "div",
                { className: "timeline-meta" },
                h("span", null, issueClipText(result, issue, index)),
                h("span", null, `evidence ${percent(issue.evidence?.confidence)}`),
              ),
            ),
          )
        : h("p", null, "No issue marker is attached to this rep."),
    ),
  );
}

function RepsTab({ result, videoSrc }) {
  const [playbackTime, setPlaybackTime] = useState(0);
  const [selectedRepId, setSelectedRepId] = useState(null);
  const replayVideoRef = useRef(null);

  useEffect(() => {
    setPlaybackTime(0);
    setSelectedRepId(null);
  }, [result?.run_id, videoSrc]);

  if (!result)
    return h(
      "section",
      { className: "summary" },
      h("h2", null, "Rep review"),
      h("p", null, "Rep segments appear after analysis."),
    );
  const report = result.report;
  const reps = report.reps?.reps || [];
  const analysisById = repAnalysisById(report);
  const timedRep = repAtTime(report, playbackTime);
  const selectedRep = reps.find((rep) => rep.rep_id === selectedRepId) || null;
  const activeRep = timedRep || selectedRep || reps[0] || null;

  function playFromRep(rep) {
    const startTime = Math.max(0, (rep.start_sec || 0) - 0.04);
    setSelectedRepId(rep.rep_id);
    setPlaybackTime(startTime);
    if (!replayVideoRef.current) return;
    replayVideoRef.current.currentTime = startTime;
    const playPromise = replayVideoRef.current.play();
    if (playPromise?.catch) playPromise.catch(() => undefined);
  }

  return h(
    "section",
    { className: "summary timeline-panel" },
    h("h2", null, "Rep replay timeline"),
    h(
      "p",
      null,
      "Play the set to follow each rep live, or click a rep segment to jump the video to that moment.",
    ),
    h(
      "div",
      { className: "replay-layout" },
      h(
        "div",
        { className: "replay-video-card" },
        videoSrc
          ? h("video", {
              className: "timeline-video",
              controls: true,
              muted: true,
              playsInline: true,
              preload: "metadata",
              ref: replayVideoRef,
              src: videoSrc,
              onLoadedMetadata: (event) => {
                setPlaybackTime(event.currentTarget.currentTime || 0);
              },
              onTimeUpdate: (event) => {
                setPlaybackTime(event.currentTarget.currentTime || 0);
              },
              onSeeked: (event) => {
                setPlaybackTime(event.currentTarget.currentTime || 0);
              },
            })
          : h(
              "div",
              { className: "timeline-video-placeholder" },
              h("strong", null, "No replay video available"),
              h("span", null, "Upload a clip or use a run with an annotated video to sync the timeline."),
            ),
        h(RepTimeline, {
          report,
          activeRepId: activeRep?.rep_id || null,
          playbackTime,
          onRepSelect: playFromRep,
        }),
      ),
      h(RepDetailPanel, { report, rep: activeRep, result, playbackTime }),
    ),
    reps.length
      ? h(
          "div",
          { className: "rep-grid" },
          reps.map((rep) => {
            const analysis = analysisById.get(rep.rep_id);
            const repIssues = issuesForRep(report, rep.rep_id);
            const score = repScore(analysis);
            return h(
              "article",
              { className: `rep-card ${scoreTone(score)}`, key: rep.rep_id },
              h(
                "div",
                { className: "rep-card-head" },
                h("strong", null, `Rep ${rep.rep_id}`),
                metaChip(score === null ? "no score" : percent(score), scoreTone(score)),
              ),
              h("span", null, timeRange(rep.start_sec, rep.end_sec)),
              h("span", null, `frames ${rep.start_frame}-${rep.end_frame}`),
              h("span", null, `midpoint ${rep.mid_sec.toFixed(2)}s`),
              h(
                "div",
                { className: "pill-row compact" },
                repIssues.length
                  ? repIssues.map((issue, index) =>
                      metaChip(issueTitle(issue), `${severityLevel(issue)}`, `${issue.issue}-${index}`),
                    )
                  : metaChip("no issue marker", "strong"),
              ),
            );
          }),
        )
      : h("p", null, "No complete reps were detected."),
  );
}

function issueEvidence(issue) {
  const degreeEntry = Object.entries(issue.evidence || {}).find(
    ([key, value]) =>
      key.endsWith("_deg") && typeof value === "number" && !Number.isNaN(value),
  );
  if (degreeEntry) {
    return `${label(degreeEntry[0].replace("_deg", ""))} about ${Math.round(degreeEntry[1])} deg`;
  }

  const cues = {
    shallow_depth: "Lower the hips a little more before standing up",
    hip_sag: "Keep hips in line with shoulders and ankles",
    incomplete_depth: "Bend deeper at the bottom of the rep",
    knee_valgus: "Keep knees tracking over the toes",
    excessive_torso_lean: "Keep the chest taller through the bottom",
    incomplete_lockout: "Finish by straightening the elbows",
    asymmetry: "Keep both sides moving evenly",
  };
  return cues[issue.issue] || "Review this part of the rep";
}

function issueTitle(issue) {
  const titles = {
    shallow_depth: "Squat depth is shallow",
    hip_sag: "Hips are dropping",
    incomplete_depth: "Rep is not deep enough",
    knee_valgus: "Knees are caving inward",
    excessive_torso_lean: "Torso leans too far forward",
    incomplete_lockout: "Lockout is incomplete",
    asymmetry: "Left and right sides are uneven",
  };
  return titles[issue.issue] || label(issue.issue);
}

function issueFocus(issue) {
  const focus = {
    shallow_depth: "Focus on hips and knees",
    hip_sag: "Focus on trunk and hips",
    incomplete_depth: "Focus on shoulders, elbows, and wrists",
    knee_valgus: "Focus on knees and ankles",
    excessive_torso_lean: "Focus on chest and hips",
    incomplete_lockout: "Focus on elbows and wrists",
    asymmetry: "Focus on left-right balance",
  };
  if (focus[issue.issue]) return focus[issue.issue];
  return `Focus on ${issue.affected_joints.map(label).join(", ")}`;
}

function severityText(issue) {
  const severity = Math.round(issue.severity * 100);
  if (severity >= 70) return `high attention ${severity}%`;
  if (severity >= 40) return `moderate attention ${severity}%`;
  return `minor attention ${severity}%`;
}

function severityLevel(issue) {
  const severity = Math.round(issue.severity * 100);
  if (severity >= 70) return "high";
  if (severity >= 40) return "moderate";
  return "minor";
}

function thumbnailForIssue(result, issue, index) {
  const thumbnails = result?.issue_thumbnail_urls || [];
  return (
    thumbnails.find(
      (thumbnail) =>
        thumbnail.rep_id === issue.rep_id && thumbnail.issue === issue.issue,
    ) || thumbnails[index]
  );
}

function clipForIssue(result, issue, index) {
  const clips = result?.issue_clip_urls || [];
  return (
    clips.find(
      (clip) => clip.rep_id === issue.rep_id && clip.issue === issue.issue,
    ) || clips[index]
  );
}

function IssueMedia({ result, issue, index }) {
  const clip = clipForIssue(result, issue, index);
  if (clip?.url) {
    return h("video", {
      className: "issue-clip",
      src: clip.url,
      controls: true,
      muted: true,
      playsInline: true,
      preload: "metadata",
    });
  }

  const thumbnail = thumbnailForIssue(result, issue, index);
  if (thumbnail?.url) {
    return h("img", {
      className: "issue-thumb",
      src: thumbnail.url,
      alt: `${label(issue.issue)} thumbnail`,
    });
  }

  return h("div", { className: "issue-thumb empty" }, "No clip");
}

function issueClipText(result, issue, index) {
  const clip = clipForIssue(result, issue, index);
  if (
    !clip ||
    typeof clip.clip_start_sec !== "number" ||
    typeof clip.clip_end_sec !== "number"
  ) {
    return `Rep ${issue.rep_id} · ${issue.start_sec.toFixed(2)}s-${issue.end_sec.toFixed(2)}s`;
  }
  return `Rep ${issue.rep_id} · clip ${clip.clip_start_sec.toFixed(2)}s-${clip.clip_end_sec.toFixed(2)}s`;
}

function IssuesTab({ result }) {
  const issues = result?.report?.issue_markers?.issues || [];
  return h(
    "section",
    { className: "summary issue-board", "aria-label": "Issue timeline" },
    h(
      "div",
      { className: "timeline-head" },
      h("h3", null, "Coaching moments"),
      h(
        "span",
        null,
        issues.length
          ? `${issues.length} interval${issues.length === 1 ? "" : "s"}`
          : "clear",
      ),
    ),
    issues.length
      ? h(
          "div",
          { className: "issue-card-grid" },
          issues.map((issue, index) =>
            h(
              "article",
              {
                className: `issue-card issue-card-feature ${severityLevel(issue)}`,
                key: `${issue.rep_id}-${issue.issue}-${index}`,
              },
              h(IssueMedia, { result, issue, index }),
              h(
                "div",
                { className: "issue-body" },
                h(
                  "div",
                  { className: "issue-card-head" },
                  h(
                    "span",
                    { className: `severity-chip ${severityLevel(issue)}` },
                    severityText(issue),
                  ),
                  h("strong", null, issueTitle(issue)),
                  h("small", null, issueClipText(result, issue, index)),
                ),
                h("p", null, issueEvidence(issue)),
                h(
                  "div",
                  { className: "timeline-meta" },
                  h(
                    "span",
                    null,
                    `issue ${timeRange(issue.start_sec, issue.end_sec)}`,
                  ),
                  h(
                    "span",
                    null,
                    `evidence ${percent(issue.evidence?.confidence)}`,
                  ),
                  h("span", null, issueFocus(issue)),
                  issue.affected_joints.length
                    ? h(
                        "span",
                        null,
                        `joints ${issue.affected_joints.map(label).join(", ")}`,
                      )
                    : null,
                ),
              ),
            ),
          ),
        )
      : h("p", null, "No sustained threshold violations were found."),
  );
}

function CoachTab({ result }) {
  if (!result)
    return h(
      "section",
      { className: "summary" },
      h("h2", null, "Coach summary"),
      h("p", null, "Coach notes appear after analysis."),
    );
  const summary = result.report.coach_summary;
  const coachArtifacts = result.report.artifacts || {};
  const coachSource = coachArtifacts.coach_summary_source || "";
  const coachProvider = coachArtifacts.coach_summary_provider || "n/a";
  const coachModel = coachArtifacts.coach_summary_model || "n/a";
  const verifierBypassed = Boolean(
    coachArtifacts.coach_summary_verifier_bypassed,
  );
  const isFallback = coachSource.startsWith("fallback");
  const cueNow = summary.top_fixes?.[0] || summary.next_session_plan?.[0] || summary.summary;
  return h(
    "section",
    { className: "summary coach-panel" },
    h("h2", null, "Coach plan"),
    h("p", null, summary.summary),
    h(
      "div",
      { className: "coach-meta pill-row" },
      [
        ["Provider", coachProvider],
        ["Model", coachModel],
        ["Source", coachSource || "n/a"],
      ].map(([name, value]) => metaChip(`${name}: ${value}`, "", name)),
    ),
    isFallback
      ? h("p", { className: "system-note" }, "A conservative fallback summary was used because the generated summary was unavailable or did not pass verification.")
      : null,
    verifierBypassed
      ? h("p", { className: "system-note warning" }, "Verifier bypass is active, so the model summary is shown even though verification did not pass.")
      : null,
    h(
      "div",
      { className: "coach-plan-grid" },
      h(BriefTile, {
        eyebrow: "Cue now",
        title: "First rep of the next set",
        body: cueNow,
        tone: "volt",
      }),
      h(NoteList, { title: "Keep", items: summary.what_looked_good, className: "coach-note" }),
      h(NoteList, { title: "Fix first", items: summary.top_fixes, className: "coach-note" }),
      h(NoteList, {
        title: "Variation vs issue",
        items: summary.valid_variation_vs_issue,
        className: "coach-note",
      }),
      h(NoteList, {
        title: "Next session",
        items: summary.next_session_plan,
        className: "coach-note",
      }),
      h(NoteList, {
        title: "Confidence notes",
        items: summary.confidence_notes,
        className: "coach-note muted",
      }),
    ),
  );
}

function JsonTab({ result }) {
  return h(
    "pre",
    { className: "json-block" },
    result ? JSON.stringify(result.report, null, 2) : "{}",
  );
}

function ArtifactsTab({ result }) {
  const links = result?.artifact_urls || [];
  return h(
    "section",
    { className: "artifact-panel" },
    h(
      "div",
      { className: "artifact-links" },
      links.length
        ? links.map((artifact) =>
            h(
              "a",
              {
                className: "artifact-link",
                key: artifact.url,
                href: artifact.url,
                download: artifact.name,
              },
              artifact.name,
              h("span", null, "download"),
            ),
          )
        : h("p", null, "Artifacts appear after analysis."),
    ),
    h(
      "details",
      { className: "json-details" },
      h("summary", null, "Raw report JSON"),
      h(JsonTab, { result }),
    ),
  );
}

const reportTabs = [
  ["summary", "Overview"],
  ["reps", "Timeline"],
  ["metrics", "Metrics"],
  ["issues", "Issues"],
  ["coach", "Coach Plan"],
  ["artifacts", "Artifacts"],
];

function ReportPanel({ result, activeTab, onTabChange, videoSrc }) {
  const content = {
    summary: h(SummaryTab, { result }),
    metrics: h(MetricsTab, { result }),
    reps: h(RepsTab, { result, videoSrc }),
    issues: h(IssuesTab, { result }),
    coach: h(CoachTab, { result }),
    artifacts: h(ArtifactsTab, { result }),
  }[activeTab];

  return h(
    React.Fragment,
    null,
    h(
      "div",
      { className: "tabs", role: "tablist", "aria-label": "Report sections" },
      reportTabs.map(([key, name]) =>
        h(
          "button",
          {
            className: `tab${activeTab === key ? " active" : ""}`,
            "aria-selected": activeTab === key,
            key,
            onClick: () => onTabChange(key),
            role: "tab",
            type: "button",
          },
          name,
        ),
      ),
    ),
    content,
  );
}

function NoteList({ title, items, className = "" }) {
  return h(
    "article",
    { className: `note${className ? ` ${className}` : ""}` },
    h("h3", null, title),
    h(
      "ul",
      null,
      items.map((item) => h("li", { key: item }, item)),
    ),
  );
}

function App() {
  const [config, setConfig] = useState(defaults);
  const [file, setFile] = useState(null);
  const [goal, setGoal] = useState("beginner_practice");
  const [experience, setExperience] = useState("beginner");
  const [exercise, setExercise] = useState("auto");
  const [equipment, setEquipment] = useState("bodyweight");
  const [limitations, setLimitations] = useState([]);
  const [result, setResult] = useState(null);
  const [activeTab, setActiveTab] = useState("summary");
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [progressSteps, setProgressSteps] = useState([]);

  const previewUrl = useMemo(
    () => (file ? URL.createObjectURL(file) : ""),
    [file],
  );

  useEffect(() => {
    fetch("/api/config")
      .then((response) => response.json())
      .then(setConfig)
      .catch(() => setConfig(defaults));
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  function toggleLimitation(value) {
    setLimitations((current) =>
      current.includes(value)
        ? current.filter((item) => item !== value)
        : [...current, value],
    );
  }

  async function analyze(event) {
    event.preventDefault();
    setError("");
    setStatus("running");
    setResult(null);
    setProgressSteps(pendingProgressState());

    const payload = new FormData();
    if (file) payload.append("video", file);
    payload.append("goal", goal);
    payload.append("experience_level", experience);
    payload.append("intended_exercise", exercise);
    payload.append("intended_variation", "");
    payload.append("limitations", JSON.stringify(limitations));
    payload.append("equipment", equipment);
    payload.append("bypass_verifier", "false");

    try {
      const response = await fetch("/api/analyze/stream", {
        method: "POST",
        body: payload,
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || "Analysis failed.");
      }
      const body = await readAnalysisStream(response, (progressEvent) => {
        setProgressSteps((currentSteps) =>
          applyProgressEvent(currentSteps, progressEvent),
        );
      });
      setResult(body);
      setProgressSteps(finalProgressState(body));
      setStatus("complete");
    } catch (caught) {
      setError(caught.message || "Analysis failed.");
      setProgressSteps([
        {
          id: "error",
          status: "active",
          text: "The scan did not finish. Try another video or check the connection, and we can run it again.",
        },
      ]);
      setStatus("idle");
    }
  }

  return h(
    "main",
    { className: "app" },
    h(
      "header",
      { className: "hero" },
      h(
        "div",
        { className: "hero-content" },
        h(
          "p",
          { className: "eyebrow" },
          "Pose intelligence for coached training",
        ),
        h("h1", null, "Pozify"),
        h("p", null, config.description),
        h(
          "div",
          { className: "hero-actions", "aria-label": "Demo strengths" },
          h("span", null, "Realtime stream"),
          h("span", null, "Annotated replay"),
          h("span", null, "Grounded coach notes"),
        ),
      ),
      h(
        "aside",
        { className: "hero-lab", "aria-label": "Motion analysis preview" },
        h(KineticFigure, null),
        h(SignalStack, null),
        h(
          "div",
          { className: "hero-metrics", "aria-label": "Pipeline highlights" },
          h(
            "div",
            { className: "metric" },
            h("strong", null, "17"),
            h("span", null, "pose landmarks"),
          ),
          h(
            "div",
            { className: "metric" },
            h("strong", null, "60s"),
            h("span", null, "clip ceiling"),
          ),
          h(
            "div",
            { className: "metric" },
            h("strong", null, "JSON"),
            h("span", null, "audit trail"),
          ),
        ),
      ),
    ),
    h(
      "form",
      { className: "workspace", onSubmit: analyze },
      h(
        "section",
        { className: "panel" },
        h(
          "div",
          { className: "panel-head" },
          h(
            "div",
            null,
            h("h2", null, "Session setup"),
            h("p", null, "Movement context"),
          ),
          h(
            "span",
            { className: "status-pill" },
            status === "running" ? "Analyzing" : result ? "Complete" : "Ready",
          ),
        ),
        h(
          "label",
          { className: "dropzone" },
          h("input", {
            name: "video",
            type: "file",
            accept: "video/*",
            onChange: (event) => setFile(event.target.files?.[0] || null),
          }),
          previewUrl
            ? h("video", {
                className: "dropzone-preview",
                src: previewUrl,
                controls: true,
              })
            : h(
                "span",
                { className: "dropzone-empty" },
                h("span", { className: "upload-icon" }, "↑"),
                h("strong", null, "Drop a workout clip"),
                h("span", null, "or click to upload an MP4, MOV, or WebM file"),
              ),
        ),
        h(
          "div",
          { className: "form-grid" },
          h(SelectField, {
            labelText: "Goal",
            name: "goal",
            value: goal,
            onChange: setGoal,
            options: config.goals,
          }),
          h(SelectField, {
            labelText: "Experience",
            name: "experience_level",
            value: experience,
            onChange: setExperience,
            options: config.experience_levels,
          }),
          h(SelectField, {
            labelText: "Exercise",
            name: "intended_exercise",
            value: exercise,
            onChange: setExercise,
            options: config.exercises,
          }),
          h(SelectField, {
            labelText: "Equipment",
            name: "equipment",
            value: equipment,
            onChange: setEquipment,
            options: config.equipment,
          }),
          h(
            "div",
            { className: "field full" },
            h("span", { className: "label" }, "Known limitations"),
            h(
              "div",
              { className: "check-grid" },
              config.limitations.map((item) =>
                h(
                  "label",
                  { className: "check-chip", key: item },
                  h("input", {
                    name: "limitations",
                    type: "checkbox",
                    checked: limitations.includes(item),
                    onChange: () => toggleLimitation(item),
                  }),
                  h("span", null, label(item)),
                ),
              ),
            ),
          ),
        ),
        h(
          "button",
          { className: "primary", disabled: status === "running", type: "submit" },
          status === "running" ? "Analyzing…" : "Analyze Form",
        ),
        error ? h("div", { className: "error" }, error) : null,
      ),
      h(
        "section",
        { className: "panel" },
        h(
          "div",
          { className: "panel-head" },
          h(
            "div",
            null,
            h("h2", null, "Review output"),
            h("p", null, "Annotated movement"),
          ),
          result
            ? h("span", { className: "status-pill" }, result.run_id)
            : null,
        ),
        h(
          "div",
          { className: "result-stage" },
          result?.annotated_video_url
            ? h("video", {
                className: "result-video",
                src: result.annotated_video_url,
                controls: true,
              })
            : h(StageEmpty, null),
        ),
        result?.annotated_video_url
          ? null
          : h(ProgressPanel, { steps: progressSteps }),
        h(ReviewInsights, { result }),
      ),
    ),
    h(ReportPanel, {
      result,
      activeTab,
      onTabChange: setActiveTab,
      videoSrc: result?.annotated_video_url || previewUrl,
    }),
  );
}

createRoot(document.getElementById("root")).render(h(App));
