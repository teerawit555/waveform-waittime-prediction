import { Search, Waves, X } from 'lucide-react';
import { toFileUrl } from '../lib/api';

type WaveformGalleryProps = {
  analysisItems: any[];
  displayedAnalysis: any[];
  totalWaves: number;
  gallerySearch: string;
  setGallerySearch: (value: string) => void;
  searchedItem: any;
  searchLoading: boolean;
  searchError: string | null;
  predictJobId: string | null;
  handleWaveSearch: () => void;
  clearSearch: () => void;
  clearSearchError: () => void;
};

export default function WaveformGallery({
  analysisItems,
  displayedAnalysis,
  totalWaves,
  gallerySearch,
  setGallerySearch,
  searchedItem,
  searchLoading,
  searchError,
  predictJobId,
  handleWaveSearch,
  clearSearch,
  clearSearchError,
}: WaveformGalleryProps) {
  return (
    <section className="card">
      <div className="section-header">
        <div className="section-title"><Waves size={18} /><span>Waveform Analysis Gallery</span></div>
      </div>

      <p className="gallery-text">
        Showing the first 30 waves by default. Search by wave_id to load any wave on demand.
      </p>

      {analysisItems.length > 0 && (
        <div className="gallery-search-wrap">
          <div className="gallery-search-input-wrap">
            <Search size={14} />
            <input
              className="gallery-search"
              type="text"
              placeholder="e.g. 1000"
              value={gallerySearch}
              onChange={(e) => {
                setGallerySearch(e.target.value);
                clearSearchError();
              }}
              onKeyDown={(e) => { if (e.key === 'Enter') handleWaveSearch(); }}
            />
          </div>

          <button
            className="primary-btn gallery-search-button"
            onClick={handleWaveSearch}
            disabled={searchLoading || !gallerySearch.trim() || !predictJobId}
          >
            {searchLoading ? 'Loading...' : 'Search'}
          </button>

          {gallerySearch && (
            <button className="gallery-search-clear" onClick={clearSearch}>
              <X size={12} className="gallery-clear-icon" />
              Clear
            </button>
          )}

          <span className="gallery-count">
            <strong>{totalWaves}</strong> waves total
          </span>
        </div>
      )}

      {searchError && (
        <div className="error-banner gallery-error">{searchError}</div>
      )}

      {searchedItem && (
        <div className="search-result-wrap">
          <div className="note-box search-result-note">
            Search result for <strong>{searchedItem.wave_id}</strong>
          </div>
          <div className="analysis-grid search-result-grid">
            <div className="analysis-card">
              <img src={toFileUrl(searchedItem.image)} alt={searchedItem.wave_id} />
              <div className="analysis-meta">
                <strong>{searchedItem.wave_id}</strong>
                <span>Pred: {Number(searchedItem.pred ?? 0).toFixed(6)}</span>
                {searchedItem.true != null && (
                  <span>True: {Number(searchedItem.true).toFixed(6)}</span>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="analysis-grid">
        {analysisItems.length === 0 ? (
          <div className="empty-state empty-action-state waveform-empty-state">
            <strong>No waveform plots yet</strong>
            <span>Run prediction on a CSV file first. The gallery will show waveform-level plots once analysis artifacts are exported.</span>
            <button type="button" className="empty-state-action" disabled>
              Waiting for prediction artifacts
            </button>
          </div>
        ) : (
          displayedAnalysis.map((item: any) => (
            <div className="analysis-card" key={`${item.wave_id}-${item.image}`}>
              <img src={toFileUrl(item.image)} alt={item.wave_id} />
              <div className="analysis-meta">
                <strong>{item.wave_id}</strong>
                <span>Pred: {Number(item.pred ?? 0).toFixed(6)}</span>
                {item.true != null && (
                  <span>True: {Number(item.true).toFixed(6)}</span>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </section>
  );
}
