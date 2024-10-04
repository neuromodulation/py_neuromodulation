import { useEffect, useRef } from "react";
import { useSocketStore } from "@/stores";
import { newPlot, react } from "plotly.js-basic-dist-min";

export const RawDataGraph = ({
  title = "Raw Data",
  xAxisTitle = "Sample",
  yAxisTitle = "Value",
  lineColor = "#1a73e8",
  maxDataPoints = 1000,
}) => {
  const graphData = useSocketStore((state) => state.graphData);
  const graphRef = useRef(null);
  const plotlyRef = useRef(null);
  const frameIdRef = useRef(null);

  const layoutRef = useRef({
    title: {
      text: title,
      font: { color: "#f4f4f4" },
    },
    autosize: true,
    height: 400,
    paper_bgcolor: "#333",
    plot_bgcolor: "#333",
    xaxis: {
      title: {
        text: xAxisTitle,
        font: { color: "#f4f4f4" },
      },
      color: "#cccccc",
    },
    yaxis: {
      title: {
        text: yAxisTitle,
        font: { color: "#f4f4f4" },
      },
      color: "#cccccc",
    },
    margin: {
      l: 50,
      r: 50,
      b: 50,
      t: 50,
    },
    font: {
      color: "#f4f4f4",
    },
  });

  // dead code?
  const updateGraph = (newData) => {
    if (!plotlyRef.current) return;

    const updatedY = newData;

    dataRef.current[0] = { ...dataRef.current[0], y: updatedY };
    react(plotlyRef.current, dataRef.current, layoutRef.current);
  };

  useEffect(() => {
    if (graphRef.current && !plotlyRef.current) {
      newPlot(
        graphRef.current,
        [
          {
            y: Array(maxDataPoints).fill(0),
            type: "scatter",
            mode: "lines",
            line: { simplify: false, color: lineColor },
          },
        ],
        layoutRef.current,
        {
          responsive: true,
          displayModeBar: false,
          // staticPlot: true,
        }
      ).then((gd) => {
        plotlyRef.current = gd;
      });
    }

    return () => {
      if (frameIdRef.current) {
        cancelAnimationFrame(frameIdRef.current);
      }
    };
  }, []);

  useEffect(() => {
    console.log("useEffect anim frame");
    if (graphData.length > 0) {
      if (!frameIdRef.current) {
        frameIdRef.current = requestAnimationFrame(updateGraph);
      }
    }
  }, [graphData, maxDataPoints, updateGraph]);

  return <div ref={graphRef}></div>;
};
