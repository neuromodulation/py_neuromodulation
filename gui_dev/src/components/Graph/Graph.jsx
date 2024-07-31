import { useEffect, useState, useRef } from "react";
import { useSocketStore } from "@/stores";
import Plotly from "plotly.js-basic-dist-min";
import styles from "./Graph.module.css";

// Plotly documentation: https://plotly.com/javascript/plotlyjs-function-reference/

export const Graph = ({
  title = "EEG Data",
  xAxisTitle = "Sample",
  yAxisTitle = "Value",
  lineColor = "blue",
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
    title: title,
    autosize: true,
    height: 400,
    xaxis: { title: xAxisTitle },
    yaxis: { title: yAxisTitle },
  });

  const updateGraph = (newData) => {
    if (!plotlyRef.current) return;

    // TODO: show a dynamic window of the last maxDataPoints
    // const updatedY = [...dataRef.current[0].y.slice(-maxDataPoints + 1), newData];
    const updatedY = newData;

    dataRef.current[0] = { ...dataRef.current[0], y: updatedY };
    Plotly.react(plotlyRef.current, dataRef.current, layoutRef.current);

    // Plotly.animate(
    //   plotlyRef.current,
    //   {
    //     data: [{ y: newData }],
    //     traces: [0],
    //     layout: {},
    //   },
    //   {
    //     transition: {
    //       duration: 32,
    //       easing: "cubic-in-out",
    //     },
    //     frame: {
    //       duration: 32,
    //     },
    //   }
    // );
  };

  // const handleNewBatch = (event) => {
  //   try {
  //     console.log("Received new batch of data");
  //     const processedData = Array.from(new Float64Array(event.data));
  //     updateGraph(processedData);
  //   } catch (err) {
  //     console.error("Error processing new data:", err);
  //   }
  // };

  useEffect(() => {
    // Initialize plot after component mount
    if (graphRef.current && !plotlyRef.current) {
      Plotly.newPlot(graphRef.current, dataRef.current, layoutRef.current).then(
        (gd) => {
          plotlyRef.current = gd;
        }
      );
    }
  });

  useEffect(() => {
    if (plotlyRef.current && graphData.length > 0) {
      dataRef.current[0] = { ...dataRef.current[0], y: graphData };
      Plotly.react(plotlyRef.current, dataRef.current, layoutRef.current);
    }
  }, [graphData]);

  useEffect(() => {
    if (!plotlyRef.current) return;

    layoutRef.current = {
      ...layoutRef.current,
      title: title,
      xaxis: { ...layoutRef.current.xaxis, title: xAxisTitle },
      yaxis: { ...layoutRef.current.yaxis, title: yAxisTitle },
    };
    Plotly.react(plotlyRef.current, dataRef, layoutRef.current);
  }, [title, xAxisTitle, yAxisTitle, lineColor]);

  return <div ref={graphRef} className={styles.graphContainer}></div>;
};
