"use client";

import { useState, useMemo } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const CLUSTER_COLORS = [
  "#6366f1", // Indigo
  "#10b981", // Emerald
  "#f59e0b", // Amber
  "#3b82f6", // Blue
  "#ec4899", // Pink
  "#8b5cf6", // Violet
  "#14b8a6", // Teal
  "#ef4444", // Red
];

// JD Star shape
function StarShape({ cx, cy }) {
  const r = 10, ir = 4.5, pts = 5;
  const coords = [];
  for (let i = 0; i < pts * 2; i++) {
    const angle = (Math.PI / pts) * i - Math.PI / 2;
    const radius = i % 2 === 0 ? r : ir;
    coords.push(`${cx + radius * Math.cos(angle)},${cy + radius * Math.sin(angle)}`);
  }
  return (
    <polygon
      points={coords.join(" ")}
      fill="#4f46e5"
      stroke="#ffffff"
      strokeWidth={2}
      filter="drop-shadow(0 2px 4px rgba(79, 70, 229, 0.4))"
    />
  );
}

// Dot Renderer
function CustomDot({ cx, cy, payload, fill, isTop10, isSelected, onPointClick }) {
  const isCandidateTop10 = isTop10(payload.id);

  if (isCandidateTop10) {
    return (
      <circle
        cx={cx}
        cy={cy}
        r={6}
        fill={fill}
        stroke="#ffffff"
        strokeWidth={2}
        style={{ cursor: "pointer", filter: "drop-shadow(0 1px 2px rgba(0,0,0,0.2))" }}
        onClick={() => onPointClick && onPointClick(payload)}
      />
    );
  }

  return (
    <circle
      cx={cx}
      cy={cy}
      r={isSelected ? 4.5 : 3}
      fill={fill}
      fillOpacity={isSelected ? 0.85 : 0.45}
      style={{ cursor: "pointer" }}
      onClick={() => onPointClick && onPointClick(payload)}
    />
  );
}

// Custom Tooltip
function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  if (!d) return null;

  return (
    <div
      style={{
        backgroundColor: "var(--bg-surface)",
        border: "1px solid var(--border-default)",
        borderRadius: "var(--radius-md)",
        padding: "0.5rem 0.75rem",
        boxShadow: "var(--shadow-dropdown)",
        fontSize: "0.75rem",
        color: "var(--text-primary)",
      }}
    >
      <div style={{ fontWeight: 700, marginBottom: "0.25rem", color: "var(--primary-text)" }}>
        Candidate #{d.id}
      </div>
      <div><strong>Cluster:</strong> {d.cluster}</div>
      {d.category && <div><strong>Category:</strong> {d.category}</div>}
      <div style={{ color: "var(--text-muted)", fontSize: "0.7rem", marginTop: "0.25rem" }}>
        Click to inspect in side panel
      </div>
    </div>
  );
}

export default function ClusterPlot({ data, selectedCluster, onSelectCluster, onSelectPoint }) {
  const {
    clusters = [],
    points = [],
    jd_point,
    top10_ids = [],
  } = data || {};

  const top10Set = useMemo(() => new Set((top10_ids || []).map(String)), [top10_ids]);
  const isTop10 = (id) => top10Set.has(String(id));

  // Group points by cluster
  const byCluster = useMemo(() => {
    const map = {};
    for (const pt of points) {
      if (!map[pt.cluster]) map[pt.cluster] = [];
      map[pt.cluster].push(pt);
    }
    return map;
  }, [points]);

  return (
    <div style={{ width: "100%", height: "460px", position: "relative" }}>
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
          <XAxis dataKey="x" type="number" hide domain={["auto", "auto"]} />
          <YAxis dataKey="y" type="number" hide domain={["auto", "auto"]} />
          <ZAxis range={[30, 30]} />
          <Tooltip content={<CustomTooltip />} />

          {Object.entries(byCluster).map(([cId, pts]) => {
            const cNum = Number(cId);
            const color = CLUSTER_COLORS[cNum % CLUSTER_COLORS.length];
            const isClusterSelected = selectedCluster === null || selectedCluster === cNum;
            const opacity = isClusterSelected ? 1 : 0.15;

            return (
              <Scatter
                key={cId}
                name={`cluster-${cId}`}
                data={pts}
                fill={color}
                opacity={opacity}
                isAnimationActive={false}
                shape={(props) => (
                  <CustomDot
                    {...props}
                    fill={color}
                    isTop10={isTop10}
                    isSelected={isClusterSelected}
                    onPointClick={onSelectPoint}
                  />
                )}
              />
            );
          })}

          {jd_point && (
            <Scatter
              name="jd"
              data={[{ x: jd_point.x, y: jd_point.y, id: "Target Job Description", cluster: "Target JD" }]}
              fill="#4f46e5"
              isAnimationActive={false}
              shape={StarShape}
            />
          )}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
