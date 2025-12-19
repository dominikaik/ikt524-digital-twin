import { useState, useCallback } from 'react';
import { apiService, ForecastRequest, ForecastResponse, Intervention } from '../services/api';

export interface ChartDataPoint {
  time: number;
  history?: number;
  csv_no_future?: number;
  csv_with_future?: number;
  d_t_bolus_carbs?: number;
  d_t_bolus_carbs_sleep_exercise?: number;
  [key: string]: number | undefined;
}

export interface ForecastState {
  chartData: ChartDataPoint[];
  interventions: Intervention[];
  shortTermPrediction: string;
  longTermPrediction: string;
  metadata: {
    filename: string;
    device: string;
    model: string;
  } | null;
  loading: boolean;
  error: string | null;
}

const useForecast = () => {
  const [state, setState] = useState<ForecastState>({
    chartData: [],
    interventions: [],
    shortTermPrediction: '',
    longTermPrediction: '',
    metadata: null,
    loading: false,
    error: null,
  });

  // Transform API response to chart data - FIXED VERSION
  const transformToChartData = useCallback((response: ForecastResponse): ChartDataPoint[] => {
    // Create a map to merge historical and prediction data by time
    const dataMap = new Map<number, ChartDataPoint>();
    
    // Add historical points
    response.historical.x.forEach((x, i) => {
      const time = x;
      dataMap.set(time, {
        time,
        history: response.historical.y[i],
      });
    });

    // Add prediction points - merge with existing time points or create new ones
    response.predictions.x.forEach((x, i) => {
      const time = x;
      const existingPoint = dataMap.get(time);
      
      if (existingPoint) {
        // Merge prediction data with existing historical data
        Object.entries(response.predictions.series).forEach(([key, values]) => {
          if (values && values[i] !== undefined && values[i] !== null) {
            existingPoint[key] = values[i];
          }
        });
      } else {
        // Create new point for prediction-only data
        const point: ChartDataPoint = { time };
        Object.entries(response.predictions.series).forEach(([key, values]) => {
          if (values && values[i] !== undefined && values[i] !== null) {
            point[key] = values[i];
          }
        });
        dataMap.set(time, point);
      }
    });

    // Convert map to sorted array
    const chartPoints = Array.from(dataMap.values()).sort((a, b) => a.time - b.time);
    
    console.log("Transformed chart data:", chartPoints);
    return chartPoints;
  }, []);

  // Calculate predictions from series data
  const calculatePredictions = useCallback((series: ForecastResponse['predictions']['series']) => {
    const primarySeries = series.d_t_bolus_carbs_sleep_exercise || series.d_t_bolus_carbs || series.csv_with_future;
    
    if (!primarySeries || primarySeries.length === 0) {
      return { shortTerm: '', longTerm: '' };
    }

    // Short term: 10 min ahead (step 2, since step 0 is current)
    const shortTermIndex = Math.min(2, primarySeries.length - 1);
    const shortTerm = primarySeries[shortTermIndex];
    
    // Long term: 60 min ahead (last step)
    const longTerm = primarySeries[primarySeries.length - 1];

    return {
      shortTerm: `Expected glucose: ${shortTerm.toFixed(1)} mg/dL (10 min)`,
      longTerm: `Expected glucose: ${longTerm.toFixed(1)} mg/dL (60 min)`,
    };
  }, []);

  // Main forecast function
  const generateForecast = useCallback(async (request: ForecastRequest) => {
    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      const response = await apiService.generateForecast(request);

      // Transform data
      const chartData = transformToChartData(response);
      const predictions = calculatePredictions(response.predictions.series);

      console.log("API RESPONSE:", response);
      console.log("PREDICTION SERIES:", response.predictions.series);
      console.log("HISTORICAL DATA:", response.historical);
      console.log("CHART DATA:", chartData);

      setState({
        chartData,
        interventions: response.interventions,
        shortTermPrediction: predictions.shortTerm,
        longTermPrediction: predictions.longTerm,
        metadata: response.metadata,
        loading: false,
        error: null,
      });

      return response;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to generate forecast';
      
      setState(prev => ({
        ...prev,
        loading: false,
        error: errorMessage,
      }));

      throw error;
    }
  }, [transformToChartData, calculatePredictions]);

  // Reset state
  const reset = useCallback(() => {
    setState({
      chartData: [],
      interventions: [],
      shortTermPrediction: '',
      longTermPrediction: '',
      metadata: null,
      loading: false,
      error: null,
    });
  }, []);

  // Clear error
  const clearError = useCallback(() => {
    setState(prev => ({ ...prev, error: null }));
  }, []);

  return {
    ...state,
    generateForecast,
    reset,
    clearError,
  };
};

export default useForecast;