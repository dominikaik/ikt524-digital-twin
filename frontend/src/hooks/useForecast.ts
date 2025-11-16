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

  // Transform API response to chart data
  const transformToChartData = useCallback((response: ForecastResponse): ChartDataPoint[] => {
    const chartPoints: ChartDataPoint[] = [];

    // Add historical points
    response.historical.x.forEach((x, i) => {
      chartPoints.push({
        time: x,
        history: response.historical.y[i],
      });
    });

    // Add prediction points
    response.predictions.x.forEach((x, i) => {
      const point: ChartDataPoint = { time: x };
      
      // Add all prediction series
      Object.entries(response.predictions.series).forEach(([key, values]) => {
        if (values && values[i] !== undefined) {
          point[key] = values[i];
        }
      });
      
      chartPoints.push(point);
    });

    return chartPoints;
  }, []);

  // Calculate predictions from series data
  const calculatePredictions = useCallback((series: ForecastResponse['predictions']['series']) => {
    const primarySeries = series.d_t_bolus_carbs_sleep_exercise || series.d_t_bolus_carbs || series.csv_with_future;
    
    if (!primarySeries || primarySeries.length === 0) {
      return { shortTerm: '', longTerm: '' };
    }

    // Short term: 10 min ahead (step 2, since step 0 is current)
    const shortTerm = primarySeries[2] || primarySeries[0];
    
    // Long term: 60 min ahead (last step)
    const longTerm = primarySeries[primarySeries.length - 1];

    return {
      shortTerm: `${shortTerm.toFixed(1)} mg/dL (10 min)`,
      longTerm: `${longTerm.toFixed(1)} mg/dL (60 min)`,
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