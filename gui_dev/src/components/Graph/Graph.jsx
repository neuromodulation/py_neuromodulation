import { useEffect, useState, useRef } from "react";
import Plotly from "plotly.js-basic-dist-min";
import { useSocketStore } from "@/stores";
import styles from "./Graph.module.css";

// Plotly documentation: https://plotly.com/javascript/plotlyjs-function-reference/

export const Graph = ({
  title = "EEG Data",
  xAxisTitle = "Sample",
  yAxisTitle = "Value",
  lineColor = "blue",
  maxDataPoints = 1000,
}) => {
  const socket = useSocketStore((state) => state.socket);
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

    dataRef.current[0] = { ...dataRef.current[0], y: newData };
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

  const handleNewBatch = (newData) => {
    try {
      console.log("Received new batch of data");
      const processedData = Array.from(new Float64Array(newData));
      updateGraph(processedData);
    } catch (err) {
      console.error("Error processing new data:", err);
    }
  };

  useEffect(() => {
    // Initialize plot after component mount
    if (graphRef.current && !plotlyRef.current) {
      Plotly.newPlot(graphRef.current, dataRef.current, layoutRef.current).then(
        (gd) => {
          plotlyRef.current = gd;
        }
      );
    }

    //Add callback to update graph on new data
    if (socket) {
      socket.on("new_batch", handleNewBatch);
    }

    // Clean up on unmount
    return () => {
      if (socket) {
        socket.off("new_batch", handleNewBatch);
      }
      if (plotlyRef.current) {
        Plotly.purge(plotlyRef.current);
        plotlyRef.current = null;
      }
    };
  }, [socket, handleNewBatch]);

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
