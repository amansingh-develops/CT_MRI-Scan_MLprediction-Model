import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useScanStore } from '../stores/scanStore';
import { useAuthStore } from '../stores/authStore';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';
import { Input } from '../components/ui/Input';
import { Search, Filter, Activity, ChevronRight, GitCompare, Loader2 } from 'lucide-react';

const MyScans = () => {
  const { scans, loadScans, isLoadingScans } = useScanStore();
  const { user } = useAuthStore();
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState('all'); // 'all' | 'anomaly' | 'clear'

  // Fetch scans from FastAPI on mount
  useEffect(() => {
    if (user?.uid) {
      loadScans(user.uid);
    }
  }, [user, loadScans]);

  const filteredScans = scans.filter(scan => {
    const matchesSearch = 
      (scan.label || '').toLowerCase().includes(search.toLowerCase()) || 
      (scan.scanId || scan.id || '').toLowerCase().includes(search.toLowerCase());
    
    if (statusFilter === 'anomaly') return matchesSearch && (scan.analysisState === 'anomaly' || scan.result?.hasAnomaly);
    if (statusFilter === 'clear') return matchesSearch && (scan.analysisState === 'clear' || (scan.result && !scan.result.hasAnomaly));
    return matchesSearch;
  });

  return (
    <div className="p-8 max-w-7xl mx-auto w-full">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-display font-bold text-on-surface mb-2">Scan Library</h1>
          <p className="text-on-surface-variant">View and manage your past AI analyses.</p>
        </div>
        <div className="flex items-center gap-3 w-full md:w-auto">
          <div className="relative flex-1 md:w-64">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-on-surface-variant" />
            <Input 
              placeholder="Search scans..." 
              className="pl-9"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
          <Button 
            variant="secondary" 
            className="shrink-0 px-3"
            onClick={() => {
              const filters = ['all', 'anomaly', 'clear'];
              const nextIdx = (filters.indexOf(statusFilter) + 1) % filters.length;
              setStatusFilter(filters[nextIdx]);
            }}
            title={`Filter: ${statusFilter}`}
          >
            <Filter className="w-4 h-4" />
            {statusFilter !== 'all' && <span className="ml-1 text-xs capitalize">{statusFilter}</span>}
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
        {filteredScans.map((scan) => (
          <Card key={scan.scanId || scan.id} className="flex flex-col overflow-hidden hover:shadow-lg transition-shadow group">
            <div className="p-5 flex-1">
              <div className="flex justify-between items-start mb-4">
                <div className="w-10 h-10 rounded-lg bg-surface-container-highest flex items-center justify-center text-on-surface">
                  <Activity className="w-5 h-5" />
                </div>
                {(scan.analysisState === 'anomaly' || scan.result?.hasAnomaly) && <Badge variant="warning">Anomaly</Badge>}
                {(scan.analysisState === 'clear' || (scan.result && !scan.result.hasAnomaly)) && <Badge variant="success">Clear</Badge>}
              </div>
              
              <Link to={`/results/${scan.scanId || scan.id}`} className="block group-hover:text-primary transition-colors">
                <h3 className="text-lg font-display font-semibold text-on-surface mb-1">{scan.label || scan.scanRef || scan.fileName || `Scan ${scan.scanId?.slice(-6) || ''}`}</h3>
              </Link>
              <p className="text-sm font-mono text-on-surface-variant mb-4">{scan.scanId || scan.id}</p>
              
              <div className="grid grid-cols-2 gap-y-2 text-sm">
                <div className="text-on-surface-variant">Date</div>
                <div className="text-right text-on-surface font-medium">{scan.uploadedAt || 'N/A'}</div>
                <div className="text-on-surface-variant">Slices</div>
                <div className="text-right text-on-surface font-medium">{scan.result?.totalSlices || scan.totalSlices || '--'}</div>
                <div className="text-on-surface-variant">Confidence</div>
                <div className="text-right text-on-surface font-mono">{(scan.confidence || scan.result?.confidence) ? `${(scan.confidence || scan.result?.confidence).toFixed(1)}%` : '--'}</div>
              </div>
            </div>
            
            <div className="p-3 border-t border-outline-variant/30 bg-surface-container flex gap-2">
              <Button asChild variant="ghost" size="sm" className="flex-1">
                <Link to={`/results/${scan.scanId || scan.id}`}>View Results</Link>
              </Button>
              <Button asChild variant="secondary" size="sm" className="flex-1">
                <Link to="/compare">
                  <GitCompare className="w-4 h-4 mr-2" />
                  Compare
                </Link>
              </Button>
            </div>
          </Card>
        ))}
        
        {filteredScans.length === 0 && (
          <div className="col-span-full py-12 text-center text-on-surface-variant">
            {isLoadingScans ? 'Loading your scans...' : 'No scans found. Upload a scan to get started.'}
          </div>
        )}
      </div>
    </div>
  );
};

export default MyScans;
