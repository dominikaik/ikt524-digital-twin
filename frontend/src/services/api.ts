const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

// Types
export interface CustomInputs {
    bolus: number;
    meal: number;
    exercise: number;
    sleep: boolean;
}

export interface ForecastRequest {
    filename: string;
    customInputs: CustomInputs;
    useFuture: boolean;
}

export interface HistoricalData {
    x: number[];
    y: number[];
}

export interface PredictionSeries {
    csv_no_future?: number[];
    csv_with_future?: number[];
    d_t_bolus_carbs?: number[];
    d_t_bolus_carbs_sleep_exercise?: number[];
    [key: string]: number[] | undefined;
}

export interface Predictions {
    x: number[];
    series: PredictionSeries;
}

export interface Intervention {
    type: 'success' | 'warning' | 'caution' | 'info';
    icon: string;
    scenario: string;
    message: string;
}

export interface Metadata {
    filename: string;
    device: string;
    model: string;
}

export interface ForecastResponse {
    historical: HistoricalData;
    predictions: Predictions;
    interventions: Intervention[];
    metadata: Metadata;
}

export interface FilesResponse {
    files: string[];
}

export interface HealthResponse {
    status: string;
    message: string;
}

export interface ApiError {
    error: string;
    traceback?: string;
}

// API Service Class
class ApiService {
    private baseUrl: string;

    constructor(baseUrl: string = API_BASE_URL) {
        this.baseUrl = baseUrl;
    }

    // Generic fetch wrapper with error handling
    private async fetchApi<T>(
        endpoint: string,
        options?: RequestInit
    ): Promise<T> {
        try {
        const response = await fetch(`${this.baseUrl}${endpoint}`, {
            ...options,
            headers: {
            'Content-Type': 'application/json',
            ...options?.headers,
            },
        });

        if (!response.ok) {
            const errorData: ApiError = await response.json().catch(() => ({
            error: `HTTP error! status: ${response.status}`,
            }));
            throw new Error(errorData.error || `Request failed with status ${response.status}`);
        }

        return await response.json();
        } catch (error) {
        if (error instanceof Error) {
            throw error;
        }
        throw new Error('An unexpected error occurred');
        }
    }

    // Health check
    async healthCheck(): Promise<HealthResponse> {
        return this.fetchApi<HealthResponse>('/health');
    }

    // Get available files
    async getFiles(): Promise<string[]> {
        const response = await this.fetchApi<FilesResponse>('/files');
        return response.files;
    }

    // Generate forecast
    async generateForecast(request: ForecastRequest): Promise<ForecastResponse> {
        return this.fetchApi<ForecastResponse>('/forecast', {
        method: 'POST',
        body: JSON.stringify(request),
        });
    }

    // Upload file
    async uploadFile(file: File): Promise<{ message: string; filename: string }> {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${this.baseUrl}/upload`, {
        method: 'POST',
        body: formData,
        });

        if (!response.ok) {
        const errorData = await response.json().catch(() => ({
            error: 'Upload failed',
        }));
        throw new Error(errorData.error || 'Upload failed');
        }

        return await response.json();
    }
    }

    // Export singleton instance
    export const apiService = new ApiService();

    // Export class for testing or custom instances
    export default ApiService;