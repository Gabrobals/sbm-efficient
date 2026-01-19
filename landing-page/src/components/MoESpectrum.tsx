import React from 'react';

interface ModelData {
  name: string;
  params: string;
  experts: number;
  savings: number;
  status: 'validated' | 'pending' | 'estimated';
}

const models: ModelData[] = [
  { name: 'Mixtral 8x7B', params: '46.7B', experts: 8, savings: 52.5, status: 'validated' },
  { name: 'Qwen1.5-MoE', params: '14.3B', experts: 60, savings: 32.4, status: 'validated' },
  { name: 'OLMoE 1B-7B', params: '6.9B', experts: 64, savings: 24.7, status: 'validated' },
  { name: 'DeepSeek-V3', params: '671B', experts: 256, savings: 0, status: 'pending' },
  { name: 'DBRX', params: '132B', experts: 16, savings: 0, status: 'estimated' },
];

export const MoESpectrum: React.FC = () => {
  const maxSavings = 60;
  
  return (
    <section className="py-20 bg-gradient-to-b from-slate-900 to-slate-950">
      <div className="container mx-auto px-4">
        {/* Header */}
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            <span className="bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
              Validated Across the MoE Spectrum
            </span>
          </h2>
          <p className="text-slate-400 text-lg max-w-2xl mx-auto">
            From lightweight research models to production-scale systems, 
            Adaptive-K consistently reduces compute while preserving quality.
          </p>
        </div>

        {/* Main Visualization */}
        <div className="max-w-4xl mx-auto">
          {/* Legend */}
          <div className="flex justify-center gap-6 mb-8 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-emerald-500"></div>
              <span className="text-slate-400">Validated</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-amber-500"></div>
              <span className="text-slate-400">Pending</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-slate-500"></div>
              <span className="text-slate-400">Estimated</span>
            </div>
          </div>

          {/* Chart */}
          <div className="bg-slate-800/50 rounded-2xl p-8 border border-slate-700">
            <div className="space-y-6">
              {models.map((model, index) => {
                const barWidth = model.savings > 0 ? (model.savings / maxSavings) * 100 : 5;
                const barColor = model.status === 'validated' 
                  ? 'from-emerald-500 to-cyan-500' 
                  : model.status === 'pending'
                    ? 'from-amber-500 to-orange-500'
                    : 'from-slate-500 to-slate-600';
                
                return (
                  <div 
                    key={model.name}
                    className="group"
                    style={{ animationDelay: `${index * 100}ms` }}
                  >
                    {/* Model Info */}
                    <div className="flex justify-between items-center mb-2">
                      <div className="flex items-center gap-3">
                        <span className="font-semibold text-white">{model.name}</span>
                        <span className="text-xs text-slate-500 bg-slate-700/50 px-2 py-0.5 rounded">
                          {model.params} · {model.experts} experts
                        </span>
                      </div>
                      <div className="text-right">
                        {model.savings > 0 ? (
                          <span className="text-emerald-400 font-bold text-lg">
                            {model.savings}% saved
                          </span>
                        ) : (
                          <span className="text-slate-500 text-sm">
                            {model.status === 'pending' ? 'Coming soon' : '~30% est.'}
                          </span>
                        )}
                      </div>
                    </div>
                    
                    {/* Progress Bar */}
                    <div className="h-4 bg-slate-700/50 rounded-full overflow-hidden">
                      <div 
                        className={`h-full bg-gradient-to-r ${barColor} rounded-full transition-all duration-1000 ease-out`}
                        style={{ 
                          width: `${barWidth}%`,
                          opacity: model.savings > 0 ? 1 : 0.5
                        }}
                      >
                        {model.savings > 0 && (
                          <div className="h-full w-full bg-gradient-to-r from-white/0 via-white/20 to-white/0 animate-shimmer" />
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* X-Axis */}
            <div className="flex justify-between mt-6 text-xs text-slate-500 border-t border-slate-700 pt-4">
              <span>0%</span>
              <span>15%</span>
              <span>30%</span>
              <span>45%</span>
              <span>60%</span>
            </div>
            <div className="text-center text-sm text-slate-400 mt-2">
              Compute Reduction (%)
            </div>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid md:grid-cols-3 gap-6 max-w-4xl mx-auto mt-12">
          <div className="bg-gradient-to-br from-emerald-500/10 to-emerald-500/5 rounded-xl p-6 border border-emerald-500/20">
            <div className="text-4xl font-bold text-emerald-400 mb-2">30.4%</div>
            <div className="text-slate-400">Average Compute Savings</div>
            <div className="text-slate-500 text-sm mt-1">Across validated models</div>
          </div>
          
          <div className="bg-gradient-to-br from-cyan-500/10 to-cyan-500/5 rounded-xl p-6 border border-cyan-500/20">
            <div className="text-4xl font-bold text-cyan-400 mb-2">33.3%</div>
            <div className="text-slate-400">Max Savings Achieved</div>
            <div className="text-slate-500 text-sm mt-1">Nemotron 3 Nano</div>
          </div>
          
          <div className="bg-gradient-to-br from-violet-500/10 to-violet-500/5 rounded-xl p-6 border border-violet-500/20">
            <div className="text-4xl font-bold text-violet-400 mb-2">&lt;0.5%</div>
            <div className="text-slate-400">Max Accuracy Drop</div>
            <div className="text-slate-500 text-sm mt-1">Quality preserved</div>
          </div>
        </div>

        {/* K Distribution Mini Chart */}
        <div className="max-w-2xl mx-auto mt-12">
          <div className="bg-slate-800/30 rounded-xl p-6 border border-slate-700/50">
            <h3 className="text-lg font-semibold text-white mb-4 text-center">
              How Adaptive-K Works (Mixtral Example)
            </h3>
            <div className="flex items-end justify-center gap-4 h-32">
              {[
                { k: 2, pct: 30, label: 'Simple queries' },
                { k: 4, pct: 50, label: 'Standard tasks' },
                { k: 6, pct: 15, label: 'Complex reasoning' },
                { k: 8, pct: 5, label: 'Full capacity' },
              ].map(({ k, pct, label }) => (
                <div key={k} className="flex flex-col items-center">
                  <div 
                    className="w-16 bg-gradient-to-t from-indigo-600 to-indigo-400 rounded-t-lg transition-all duration-500"
                    style={{ height: `${pct}%` }}
                  />
                  <div className="text-white font-semibold mt-2">K={k}</div>
                  <div className="text-slate-500 text-xs">{pct}%</div>
                  <div className="text-slate-600 text-xs mt-1 max-w-[80px] text-center">{label}</div>
                </div>
              ))}
            </div>
            <p className="text-center text-slate-400 text-sm mt-4">
              Most queries only need 2-4 experts, not the full 8
            </p>
          </div>
        </div>

        {/* CTA */}
        <div className="text-center mt-16">
          <a 
            href="#pricing" 
            className="inline-flex items-center gap-2 bg-gradient-to-r from-emerald-500 to-cyan-500 text-white px-8 py-4 rounded-xl font-semibold hover:opacity-90 transition-opacity"
          >
            Start Saving Compute Today
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
            </svg>
          </a>
          <p className="text-slate-500 text-sm mt-4">
            pip install adaptive-k-routing
          </p>
        </div>
      </div>

      <style jsx>{`
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        .animate-shimmer {
          animation: shimmer 2s infinite;
        }
      `}</style>
    </section>
  );
};

export default MoESpectrum;
