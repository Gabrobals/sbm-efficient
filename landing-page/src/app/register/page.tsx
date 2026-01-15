'use client';

import { useState } from 'react';
import Link from 'next/link';

export default function RegisterPage() {
  const [formData, setFormData] = useState({
    email: '',
    company: '',
    useCase: '',
    acceptTerms: false
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ success: boolean; licenseKey?: string; error?: string } | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      const response = await fetch('/api/license/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      setResult({ success: false, error: 'Registration failed. Please try again.' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 py-20">
      <div className="max-w-lg mx-auto px-4 sm:px-6 lg:px-8">
        
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-white mb-4">
            Register for Adaptive-K
          </h1>
          <p className="text-gray-400">
            Get your free Community license to start using Adaptive-K
          </p>
        </div>

        {/* Registration Form */}
        {!result?.success ? (
          <form onSubmit={handleSubmit} className="bg-gray-800 rounded-xl p-8">
            <div className="space-y-6">
              
              {/* Email */}
              <div>
                <label className="block text-gray-300 mb-2">Work Email *</label>
                <input
                  type="email"
                  required
                  value={formData.email}
                  onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                  placeholder="you@company.com"
                  className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-cyan-500"
                />
              </div>

              {/* Company */}
              <div>
                <label className="block text-gray-300 mb-2">Company / Organization *</label>
                <input
                  type="text"
                  required
                  value={formData.company}
                  onChange={(e) => setFormData({ ...formData, company: e.target.value })}
                  placeholder="Your Company Name"
                  className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-cyan-500"
                />
              </div>

              {/* Use Case */}
              <div>
                <label className="block text-gray-300 mb-2">Primary Use Case</label>
                <select
                  value={formData.useCase}
                  onChange={(e) => setFormData({ ...formData, useCase: e.target.value })}
                  className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-cyan-500"
                >
                  <option value="">Select use case...</option>
                  <option value="research">Academic Research</option>
                  <option value="startup">Startup / MVP</option>
                  <option value="enterprise">Enterprise Evaluation</option>
                  <option value="personal">Personal Project</option>
                  <option value="other">Other</option>
                </select>
              </div>

              {/* Terms */}
              <div className="flex items-start">
                <input
                  type="checkbox"
                  id="terms"
                  required
                  checked={formData.acceptTerms}
                  onChange={(e) => setFormData({ ...formData, acceptTerms: e.target.checked })}
                  className="mt-1 mr-3"
                />
                <label htmlFor="terms" className="text-gray-400 text-sm">
                  I agree to the <Link href="/terms" className="text-cyan-400 hover:underline">Terms of Service</Link> and 
                  understand that usage is tracked for analytics purposes. *
                </label>
              </div>

              {/* Error Message */}
              {result?.error && (
                <div className="bg-red-900/50 border border-red-500 rounded-lg p-4 text-red-300">
                  {result.error}
                </div>
              )}

              {/* Submit Button */}
              <button
                type="submit"
                disabled={loading}
                className="w-full bg-cyan-600 hover:bg-cyan-500 disabled:bg-gray-600 text-white font-semibold py-3 px-6 rounded-lg transition-colors"
              >
                {loading ? 'Registering...' : 'Get Free License Key'}
              </button>
            </div>

            {/* Upgrade Note */}
            <div className="mt-6 pt-6 border-t border-gray-700 text-center">
              <p className="text-gray-400 text-sm">
                Need CUDA kernels, vLLM integration, or commercial support?
              </p>
              <Link href="/#pricing" className="text-cyan-400 hover:underline text-sm">
                View Professional & Enterprise plans →
              </Link>
            </div>
          </form>
        ) : (
          /* Success */
          <div className="bg-gray-800 rounded-xl p-8">
            <div className="text-center mb-6">
              <div className="w-16 h-16 bg-green-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <h2 className="text-2xl font-bold text-white mb-2">Registration Successful!</h2>
              <p className="text-gray-400">Your Community license key is ready</p>
            </div>

            {/* License Key */}
            <div className="bg-gray-900 rounded-lg p-4 mb-6">
              <label className="block text-gray-400 text-sm mb-2">Your License Key:</label>
              <code className="block text-cyan-400 text-xs break-all font-mono">
                {result.licenseKey}
              </code>
            </div>

            {/* Instructions */}
            <div className="space-y-4 text-sm">
              <h3 className="font-semibold text-white">Quick Start:</h3>
              <div className="bg-gray-900 rounded-lg p-4">
                <code className="text-gray-300">
                  <span className="text-gray-500"># 1. Install</span><br/>
                  pip install adaptive-k-routing<br/><br/>
                  <span className="text-gray-500"># 2. Set license</span><br/>
                  export ADAPTIVE_K_LICENSE=&quot;{result.licenseKey?.substring(0, 30)}...&quot;<br/><br/>
                  <span className="text-gray-500"># 3. Use</span><br/>
                  from adaptive_k import AdaptiveKRouter
                </code>
              </div>
            </div>

            {/* Email Note */}
            <p className="text-gray-400 text-sm mt-6 text-center">
              📧 A copy of this license key has been sent to your email.
            </p>

            {/* Links */}
            <div className="flex gap-4 mt-6">
              <Link href="/portal" className="flex-1 bg-gray-700 hover:bg-gray-600 text-center py-2 rounded-lg text-white transition-colors">
                Go to Portal
              </Link>
              <Link href="/docs/INSTALLATION_GUIDE.md" className="flex-1 bg-cyan-600 hover:bg-cyan-500 text-center py-2 rounded-lg text-white transition-colors">
                View Guide
              </Link>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
