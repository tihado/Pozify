import React, {
  useEffect,
  useMemo,
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

function SelectField({ labelText, value, onChange, options }) {
  return h(
    Field,
    { labelText },
    h(
      "select",
      { value, onChange: (event) => onChange(event.target.value) },
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
  if (!response.body) throw new Error("Streaming is not available in this browser.");
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
      if (event.type === "error") throw new Error(event.detail || "Analysis failed.");
    }
  }

  if (buffer.trim()) {
    const event = JSON.parse(buffer);
    if (event.type === "progress") onEvent(event);
    if (event.type === "complete") result = event.result;
    if (event.type === "error") throw new Error(event.detail || "Analysis failed.");
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
    h("h3", null, isComplete ? "Scan results are ready" : "Your scan is moving"),
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
        warnings.length ? warnings.map(label).join(", ") : "clear enough to coach",
      ),
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
  const issues = report.issue_markers?.issues || [];
  const stats = [
    ["Exercise", report.exercise.exercise],
    ["Confidence", percent(report.exercise.confidence)],
    ["Variation", report.variation.detected_variation],
    ["Reps", String(report.reps.reps.length)],
    ["Issues", String(issues.length)],
  ];

  return h(
    "section",
    { className: "summary" },
    h("h2", null, "Scan summary"),
    h("p", null, summary.summary),
    h(
      "div",
      { className: "stat-grid" },
      stats.map(([name, value]) =>
        h(
          "div",
          { className: "stat", key: name },
          h("span", null, name),
          h("strong", null, value),
        ),
      ),
    ),
    h(
      "div",
      { className: "note-grid" },
      h(NoteList, { title: "What went well", items: summary.what_went_well }),
      h(NoteList, { title: "Top fixes", items: summary.top_fixes }),
      h(NoteList, { title: "Next session", items: summary.next_session_plan }),
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
  return h(
    "section",
    { className: "summary" },
    h("h2", null, "Movement metrics"),
    h(
      "div",
      { className: "metric-grid" },
      aggregate.length
        ? aggregate.map(([name, value]) =>
            h(
              "div",
              { className: "stat", key: name },
              h("span", null, label(name)),
              h("strong", null, formatValue(value)),
            ),
          )
        : h("p", null, "No aggregate metrics available."),
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

function RepsTab({ result }) {
  if (!result)
    return h(
      "section",
      { className: "summary" },
      h("h2", null, "Rep review"),
      h("p", null, "Rep segments appear after analysis."),
    );
  const reps = result.report.reps?.reps || [];
  return h(
    "section",
    { className: "summary" },
    h("h2", null, "Rep review"),
    reps.length
      ? h(
          "div",
          { className: "rep-grid" },
          reps.map((rep) =>
            h(
              "article",
              { className: "rep-card", key: rep.rep_id },
              h("strong", null, `Rep ${rep.rep_id}`),
              h(
                "span",
                null,
                `${rep.start_sec.toFixed(2)}s-${rep.end_sec.toFixed(2)}s`,
              ),
              h("span", null, `frames ${rep.start_frame}-${rep.end_frame}`),
              h("span", null, `midpoint ${rep.mid_sec.toFixed(2)}s`),
            ),
          ),
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
    { className: "summary", "aria-label": "Issue timeline" },
    h(
      "div",
      { className: "timeline-head" },
      h("h3", null, "Issue timeline"),
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
                className: "issue-card",
                key: `${issue.rep_id}-${issue.issue}-${index}`,
              },
              h(IssueMedia, { result, issue, index }),
              h(
                "div",
                { className: "timeline-main" },
                h("strong", null, issueTitle(issue)),
                h("span", null, issueClipText(result, issue, index)),
                h(
                  "span",
                  null,
                  `issue ${issue.start_sec.toFixed(2)}s-${issue.end_sec.toFixed(2)}s`,
                ),
              ),
              h(
                "div",
                { className: "timeline-meta" },
                h(
                  "span",
                  { className: `severity-chip ${severityLevel(issue)}` },
                  severityText(issue),
                ),
                h(
                  "span",
                  null,
                  `evidence confidence ${percent(issue.evidence?.confidence)}`,
                ),
                h("span", null, issueEvidence(issue)),
                h("span", null, issueFocus(issue)),
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
  return h(
    "section",
    { className: "summary" },
    h("h2", null, "Coach summary"),
    h("p", null, summary.summary),
    h(
      "div",
      { className: "note-grid" },
      h(NoteList, { title: "Main findings", items: summary.main_findings }),
      h(NoteList, { title: "Top fixes", items: summary.top_fixes }),
      h(NoteList, {
        title: "Confidence notes",
        items: summary.confidence_notes,
      }),
    ),
    h(
      "article",
      { className: "note full-note" },
      h("h3", null, "Variation context"),
      h("p", null, summary.variation_explanation),
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
  );
}

const reportTabs = [
  ["summary", "Summary"],
  ["metrics", "Metrics"],
  ["reps", "Reps"],
  ["issues", "Issues"],
  ["coach", "Coach"],
  ["json", "JSON"],
  ["artifacts", "Artifacts"],
];

function ReportPanel({ result, activeTab, onTabChange }) {
  const content = {
    summary: h(SummaryTab, { result }),
    metrics: h(MetricsTab, { result }),
    reps: h(RepsTab, { result }),
    issues: h(IssuesTab, { result }),
    coach: h(CoachTab, { result }),
    json: h(JsonTab, { result }),
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
            key,
            onClick: () => onTabChange(key),
            type: "button",
          },
          name,
        ),
      ),
    ),
    content,
  );
}

function NoteList({ title, items }) {
  return h(
    "article",
    { className: "note" },
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
  const [variation, setVariation] = useState("");
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
    payload.append("intended_variation", variation);
    payload.append("limitations", JSON.stringify(limitations));
    payload.append("equipment", equipment);

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
      "section",
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
      ),
      h(
        "div",
        { className: "hero-metrics", "aria-label": "Pipeline highlights" },
        h(
          "div",
          { className: "metric" },
          h("strong", null, "33"),
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
            status === "running" ? "Analyzing" : "Ready",
          ),
        ),
        h(
          "label",
          { className: "dropzone" },
          h("input", {
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
                h("strong", null, "Drop video here"),
                h("span", null, "or click to upload"),
              ),
        ),
        h(
          "div",
          { className: "form-grid" },
          h(SelectField, {
            labelText: "Goal",
            value: goal,
            onChange: setGoal,
            options: config.goals,
          }),
          h(SelectField, {
            labelText: "Experience",
            value: experience,
            onChange: setExperience,
            options: config.experience_levels,
          }),
          h(SelectField, {
            labelText: "Exercise",
            value: exercise,
            onChange: setExercise,
            options: config.exercises,
          }),
          h(SelectField, {
            labelText: "Equipment",
            value: equipment,
            onChange: setEquipment,
            options: config.equipment,
          }),
          h(
            Field,
            { labelText: "Variation", full: true },
            h("input", {
              type: "text",
              value: variation,
              placeholder: "Optional, e.g. wide_grip_push_up",
              onChange: (event) => setVariation(event.target.value),
            }),
          ),
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
          { className: "primary", disabled: status === "running" },
          status === "running" ? "Analyzing..." : "Analyze Form",
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
            : h(
                "div",
                { className: "result-empty" },
                h("strong", null, "Awaiting scan"),
                h("span", null, "The annotated video appears after analysis."),
              ),
        ),
        result?.annotated_video_url
          ? null
          : h(ProgressPanel, { steps: progressSteps }),
        h(ReviewInsights, { result }),
      ),
    ),
    h(ReportPanel, { result, activeTab, onTabChange: setActiveTab }),
  );
}

createRoot(document.getElementById("root")).render(h(App));
