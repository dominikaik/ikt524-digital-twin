import { useState, useEffect, useCallback } from 'react';
import { apiService } from '../services/api';

export interface UseFilesState {
  files: string[];
  selectedFile: string;
  loading: boolean;
  error: string | null;
}

export const useFiles = () => {
  const [state, setState] = useState<UseFilesState>({
    files: [],
    selectedFile: '',
    loading: false,
    error: null,
  });

  // Fetch files from API
  const fetchFiles = useCallback(async () => {
    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      const files = await apiService.getFiles();
      
      setState({
        files,
        selectedFile: files.length > 0 ? files[0] : '',
        loading: false,
        error: null,
      });

      return files;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to load files';
      
      setState(prev => ({
        ...prev,
        loading: false,
        error: errorMessage,
      }));

      throw error;
    }
  }, []);

  // Set selected file
  const setSelectedFile = useCallback((filename: string) => {
    setState(prev => ({ ...prev, selectedFile: filename }));
  }, []);

  // Upload new file
  const uploadFile = useCallback(async (file: File) => {
    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      const response = await apiService.uploadFile(file);
      
      // Refresh file list after upload
      await fetchFiles();
      
      // Select the newly uploaded file
      setState(prev => ({
        ...prev,
        selectedFile: response.filename,
        loading: false,
      }));

      return response;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to upload file';
      
      setState(prev => ({
        ...prev,
        loading: false,
        error: errorMessage,
      }));

      throw error;
    }
  }, [fetchFiles]);

  // Clear error
  const clearError = useCallback(() => {
    setState(prev => ({ ...prev, error: null }));
  }, []);

  // Load files on mount
  useEffect(() => {
    fetchFiles();
  }, [fetchFiles]);

  return {
    ...state,
    fetchFiles,
    setSelectedFile,
    uploadFile,
    clearError,
  };
};

export default useFiles;