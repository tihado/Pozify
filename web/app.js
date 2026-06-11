import React, { useEffect, useMemo, useState } from "https://esm.sh/react@18.2.0";
import { createRoot } from "https://esm.sh/react-dom@18.2.0/client";

const h = React.createElement;

const defaults = {
  description:
    "Upload a short workout clip, tune the athlete context, and generate an annotated form-review report with structured artifacts.",
  goals: ["strength", "hypertrophy", "endurance", "mobility", "beginner_practice"],
  experience_levels: ["beginner", "intermediate"],
  exercises: ["auto"],
  limitations: ["wrist_discomfort", "knee_discomfort", "shoulder_discomfort"],
  equipment: ["bodyweight", "dumbbell", "barbell", "unknown"],
};

function label(value) {
  return value.replaceAll("_", " ");
}

async function readResponseBody(response) {
  const text = await response.text();
  if (!text) return {};

  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) return JSON.parse(text);

  return { detail: text };
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
      options.map((option) => h("option", { key: option, value: option }, label(option))),
    ),
  );
}

function Summary({ result }) {
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
    ["Variation", report.variation.detected_variation],
    ["Reps", String(report.reps.reps.length)],
    ["Issues", String(issues.length)],
    ["Mode", report.artifacts.analysis_mode],
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
        h("div", { className: "stat", key: name }, h("span", null, name), h("strong", null, value)),
      ),
    ),
    h(
      "div",
      { className: "note-grid" },
      h(NoteList, { title: "What went well", items: summary.what_went_well }),
      h(NoteList, { title: "Top fixes", items: summary.top_fixes }),
      h(NoteList, { title: "Next session", items: summary.next_session_plan }),
    ),
    h(IssueTimeline, { issues }),
    warnings.length
      ? h(
          "div",
          { className: "quality-list" },
          warnings.map((warning) => h("span", { key: warning }, label(warning))),
        )
      : h("div", { className: "quality-list" }, h("span", null, "No quality warnings")),
  );
}

function issueEvidence(issue) {
  const entries = Object.entries(issue.evidence || {}).filter(
    ([key, value]) =>
      !["threshold", "confidence", "variation_context", "supporting_frames", "fallback"].includes(key) &&
      typeof value !== "object",
  );
  const [metric, value] = entries[0] || ["metric", "n/a"];
  return `${label(metric)} ${value} vs ${issue.evidence?.threshold ?? "n/a"}`;
}

function IssueTimeline({ issues }) {
  return h(
    "section",
    { className: "issue-timeline", "aria-label": "Issue timeline" },
    h(
      "div",
      { className: "timeline-head" },
      h("h3", null, "Issue timeline"),
      h("span", null, issues.length ? `${issues.length} interval${issues.length === 1 ? "" : "s"}` : "clear"),
    ),
    issues.length
      ? h(
          "div",
          { className: "timeline-list" },
          issues.map((issue, index) =>
            h(
              "article",
              { className: "timeline-item", key: `${issue.rep_id}-${issue.issue}-${index}` },
              h(
                "div",
                { className: "timeline-main" },
                h("strong", null, label(issue.issue)),
                h("span", null, `Rep ${issue.rep_id} · ${issue.start_sec.toFixed(2)}s-${issue.end_sec.toFixed(2)}s`),
              ),
              h(
                "div",
                { className: "timeline-meta" },
                h("span", null, `severity ${Math.round(issue.severity * 100)}%`),
                h("span", null, issueEvidence(issue)),
                h("span", null, issue.affected_joints.map(label).join(", ")),
              ),
            ),
          ),
        )
      : h("p", null, "No sustained threshold violations were found."),
  );
}

function NoteList({ title, items }) {
  return h(
    "article",
    { className: "note" },
    h("h3", null, title),
    h("ul", null, items.map((item) => h("li", { key: item }, item))),
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
  const [activeTab, setActiveTab] = useState("json");
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");

  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : ""), [file]);

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
      current.includes(value) ? current.filter((item) => item !== value) : [...current, value],
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
      const response = await fetch("/api/analyze", { method: "POST", body: payload });
      const body = await readResponseBody(response);
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
        h("p", { className: "eyebrow" }, "Pose intelligence for coached training"),
        h("h1", null, "Pozify"),
        h("p", null, config.description),
      ),
      h(
        "div",
        { className: "hero-metrics", "aria-label": "Pipeline highlights" },
        h("div", { className: "metric" }, h("strong", null, "33"), h("span", null, "pose landmarks")),
        h("div", { className: "metric" }, h("strong", null, "60s"), h("span", null, "clip ceiling")),
        h("div", { className: "metric" }, h("strong", null, "JSON"), h("span", null, "audit trail")),
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
          h("div", null, h("h2", null, "Session setup"), h("p", null, "Movement context")),
          h("span", { className: "status-pill" }, status === "running" ? "Analyzing" : "Ready"),
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
            ? h("video", { className: "dropzone-preview", src: previewUrl, controls: true })
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
          h(SelectField, { labelText: "Goal", value: goal, onChange: setGoal, options: config.goals }),
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
        h("button", { className: "primary", disabled: status === "running" }, status === "running" ? "Analyzing..." : "Analyze Form"),
        error ? h("div", { className: "error" }, error) : null,
      ),
      h(
        "section",
        { className: "panel" },
        h(
          "div",
          { className: "panel-head" },
          h("div", null, h("h2", null, "Review output"), h("p", null, "Annotated movement")),
          result ? h("span", { className: "status-pill" }, result.run_id) : null,
        ),
        h(
          "div",
          { className: "result-stage" },
          result?.annotated_video_url
            ? h("video", { className: "result-video", src: result.annotated_video_url, controls: true })
            : h(
                "div",
                { className: "result-empty" },
                h("strong", null, "Awaiting scan"),
                h("span", null, "The annotated video appears after analysis."),
              ),
        ),
      ),
    ),
    h(Summary, { result }),
    h(
      "div",
      { className: "tabs", role: "tablist", "aria-label": "Artifacts" },
      ["json", "artifacts"].map((tab) =>
        h(
          "button",
          {
            className: `tab${activeTab === tab ? " active" : ""}`,
            key: tab,
            onClick: () => setActiveTab(tab),
            type: "button",
          },
          tab === "json" ? "Final report JSON" : "Artifact links",
        ),
      ),
    ),
    h(
      "section",
      { className: "artifact-panel" },
      activeTab === "json"
        ? h("pre", { className: "json-block" }, result ? JSON.stringify(result.report, null, 2) : "{}")
        : h(
            "div",
            { className: "artifact-links" },
            result
              ? [
                  h("a", { className: "artifact-link", key: "report", href: result.final_report_url }, "final_report.json", h("span", null, "open")),
                  result.annotated_video_url
                    ? h("a", { className: "artifact-link", key: "video", href: result.annotated_video_url }, "annotated_video.mp4", h("span", null, "open"))
                    : null,
                ]
              : h("p", null, "Artifacts appear after analysis."),
          ),
    ),
  );
}

createRoot(document.getElementById("root")).render(h(App));
