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
  const entries = Object.entries(issue.evidence || {}).filter(
    ([key, value]) =>
      ![
        "threshold",
        "confidence",
        "variation_context",
        "supporting_frames",
        "fallback",
      ].includes(key) && typeof value !== "object",
  );
  const [metric, value] = entries[0] || ["metric", "n/a"];
  return `${label(metric)} ${value} vs ${issue.evidence?.threshold ?? "n/a"}`;
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
                h("strong", null, label(issue.issue)),
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
                  null,
                  `severity ${Math.round(issue.severity * 100)}%`,
                ),
                h(
                  "span",
                  null,
                  `confidence ${percent(issue.evidence?.confidence)}`,
                ),
                h("span", null, issueEvidence(issue)),
                h("span", null, issue.affected_joints.map(label).join(", ")),
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

    const payload = new FormData();
    if (file) payload.append("video", file);
    payload.append("goal", goal);
    payload.append("experience_level", experience);
    payload.append("intended_exercise", exercise);
    payload.append("intended_variation", variation);
    payload.append("limitations", JSON.stringify(limitations));
    payload.append("equipment", equipment);

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        body: payload,
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || "Analysis failed.");
      setResult(body);
      setStatus("complete");
    } catch (caught) {
      setError(caught.message || "Analysis failed.");
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
      ),
    ),
    h(ReportPanel, { result, activeTab, onTabChange: setActiveTab }),
  );
}

createRoot(document.getElementById("root")).render(h(App));
