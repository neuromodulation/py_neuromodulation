import { useEffect, useState, useRef } from "react";
import { useSocketStore } from "@/stores";
import { newPlot, react } from "plotly.js-dist-min";
import styles from "./Graph.module.css";

export const Graph = ({
  title = "EEG Data",
  xAxisTitle = "Sample",
  yAxisTitle = "Value",
  lineColor = "#1a73e8",
  maxDataPoints = 1000,
}) => {
  const { graphData } = useSocketStore();

  const graphRef = useRef(null);
  const plotlyRef = useRef(null);

  const dataRef = useRef([
    {
      y: Array(maxDataPoints).fill(0),
      type: "scatter",
      mode: "lines",
      line: { simplify: false, color: lineColor },
    },
  ]);

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

  const updateGraph = (newData) => {
    if (!plotlyRef.current) return;

    const updatedY = newData;

    dataRef.current[0] = { ...dataRef.current[0], y: updatedY };
    react(plotlyRef.current, dataRef.current, layoutRef.current);
  };

  useEffect(() => {
    if (graphRef.current && !plotlyRef.current) {
      newPlot(graphRef.current, dataRef.current, layoutRef.current).then(
        (gd) => {
          plotlyRef.current = gd;
        }
      );
    }
  });

  useEffect(() => {
    if (plotlyRef.current && graphData.length > 0) {
      dataRef.current[0] = { ...dataRef.current[0], y: graphData };
      react(plotlyRef.current, dataRef.current, layoutRef.current);
    }
  }, [graphData]);

  useEffect(() => {
    if (!plotlyRef.current) return;

    layoutRef.current = {
      ...layoutRef.current,
      title: { text: title },
      xaxis: { ...layoutRef.current.xaxis, title: { text: xAxisTitle } },
      yaxis: { ...layoutRef.current.yaxis, title: { text: yAxisTitle } },
    };
    react(plotlyRef.current, dataRef, layoutRef.current);
  }, [title, xAxisTitle, yAxisTitle, lineColor]);

  return <div ref={graphRef} className={styles.graphContainer}></div>;
};
