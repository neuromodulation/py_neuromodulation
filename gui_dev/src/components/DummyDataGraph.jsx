import React, { useEffect, useRef, useState } from "react";
import ReactECharts from 'echarts-for-react';

export const DemoChart = () => {
  const [data, setData] = useState([[Date.now(), Math.random() * 100]]);
  const chartRef = useRef(null);

  useEffect(() => {
    const timer = setInterval(() => {
      setData(prev => [...prev.slice(-50), [Date.now(), Math.random() * 100]]);
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  const option = {
    xAxis: { type: 'time' },
    yAxis: { type: 'value' },
    series: [
        {
            name: '1',
            type: 'line',
            data: data,
            showSymbol: false,
            stack: 'Total',
        },
        {
            name: '2',
            type: 'line',
            data: data,
            showSymbol: false,
            stack: 'Total',
        },
        {
            name: '3',
            type: 'line',
            data: data,
            showSymbol: false,
            stack: 'Total',
        },

    ]
  };

  // return <ReactECharts ref={chartRef} option={option} style={{ height: 400 }} />;
};
