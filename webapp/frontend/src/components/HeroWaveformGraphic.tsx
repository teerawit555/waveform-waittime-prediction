import React from 'react';

export const HeroWaveformGraphic: React.FC = () => {
  // SVG Dimensions: 600x260
  // Shifted slightly to the right to accommodate the black Y-axis line at x=50.
  // Curve starts at (55, 230), rises smoothly, and settles at y=70 starting at x=260, then flatlines to 580.
  const pathData = "M 55 230 C 115 230, 180 70, 260 70 L 580 70";
  const predX = 260;

  return (
    <div className="hero-minimal-wave">
      <svg viewBox="0 0 600 260" className="minimal-wave-svg" width="100%" height="100%">
        <defs>
          {/* Gradient for the wave line - fully opaque vibrant royal blue to glowing sky blue/cyan */}
          <linearGradient id="vibrant-blue-grad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#004ae0" stopOpacity="1" />
            <stop offset="45%" stopColor="#0066cc" stopOpacity="1" />
            <stop offset="100%" stopColor="#00c8e6" stopOpacity="1" />
          </linearGradient>

          {/* Clip path to animate the prediction line drawing itself downwards at x=260 */}
          <clipPath id="pred-line-anim-clip">
            <rect x="255" y="70" width="10" height="0" className="clip-rect-anim" />
          </clipPath>
        </defs>

        {/* Ambient background shape under the curve */}
        <path
          d="M 55 230 C 115 230, 180 70, 260 70 L 580 70 L 580 235 L 55 235 Z"
          fill="rgba(0, 74, 224, 0.04)"
          className="wave-ambient-fill"
        />

        {/* The main smooth glowing waveform line - Fined to 6px stroke-width for perfect proportion balance */}
        <path
          d={pathData}
          className="wave-line-glow"
          stroke="url(#vibrant-blue-grad)"
          strokeWidth="6"
          fill="none"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Vertical Prediction line - Animated to draw itself downwards */}
        <g clipPath="url(#pred-line-anim-clip)">
          <line
            x1={predX}
            y1="70"
            x2={predX}
            y2="235"
            className="wave-pred-dashed-line"
          />
        </g>

        {/* Sharp glowing dot at the settling point */}
        <circle
          cx={predX}
          cy="70"
          r="6.5"
          className="wave-settle-dot"
          fill="#004ae0"
        />
        <circle
          cx={predX}
          cy="70"
          r="12.5"
          className="wave-settle-pulse"
          fill="none"
          stroke="#00c8e6"
        />

        {/* Premium Corporate Navy Slate Axes (L-shape) with Solid Arrowheads & Ticks */}
        <g className="plt-axes">
          {/* Y Axis line */}
          <line x1="50" y1="35" x2="50" y2="235" className="plt-axis-line" />
          {/* Y Axis Solid Arrowhead (pointing UP) */}
          <polygon points="46,42 50,34 54,42" className="plt-axis-arrow-head" />
          
          {/* Y Ticks (Leftwards) */}
          <line x1="44" y1="194.25" x2="50" y2="194.25" className="plt-axis-tick" />
          <line x1="44" y1="153.5" x2="50" y2="153.5" className="plt-axis-tick" />
          <line x1="44" y1="112.75" x2="50" y2="112.75" className="plt-axis-tick" />
          <line x1="44" y1="72" x2="50" y2="72" className="plt-axis-tick" />

          {/* X Axis line */}
          <line x1="50" y1="235" x2="580" y2="235" className="plt-axis-line" />
          {/* X Axis Solid Arrowhead (pointing RIGHT) */}
          <polygon points="572,231 580,235 572,239" className="plt-axis-arrow-head" />
          
          {/* X Ticks (Downwards) */}
          <line x1="156" y1="235" x2="156" y2="241" className="plt-axis-tick" />
          <line x1="262" y1="235" x2="262" y2="241" className="plt-axis-tick" />
          <line x1="368" y1="235" x2="368" y2="241" className="plt-axis-tick" />
          <line x1="474" y1="235" x2="474" y2="241" className="plt-axis-tick" />
        </g>

        {/* Premium Axis Labels - ALL CAPS, bold navy, elegant typography */}
        <g className="plt-axis-labels">
          {/* Y label: VALUE (positioned centered above the top Y arrowhead) */}
          <text x="50" y="18" className="plt-axis-text" textAnchor="middle">
            VALUE
          </text>
          {/* X label: TIME (positioned below the end X arrowhead) */}
          <text x="580" y="256" className="plt-axis-text" textAnchor="end">
            TIME
          </text>
        </g>
      </svg>
    </div>
  );
};
