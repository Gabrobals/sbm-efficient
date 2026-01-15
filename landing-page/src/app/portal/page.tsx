'use client';

import Link from 'next/link';
import { useState } from 'react';

interface LicenseInfo {
  company: string;
  tier: string;
  expires: string;
  email: string;
  issued: string;
  valid: boolean;
  daysRemaining: number;
}

function validateLicense(licenseKey: string): LicenseInfo | null {
  try {
    const [payloadB64, signature] = licenseKey.split('.');
    if (!payloadB64 || !signature) return null;
    
    // Decode payload
    const payloadJson = atob(payloadB64);
    const payload = JSON.parse(payloadJson);
    
    // Check expiration
    const expiresDate = new Date(payload.expires);
    const now = new Date();
    const daysRemaining = Math.ceil((expiresDate.getTime() - now.getTime()) / (1000 * 60 * 60 * 24));
    
    return {
      company: payload.company || 'Unknown',
      tier: payload.tier || 'unknown',
      expires: payload.expires,
      email: payload.email || '',
      issued: payload.issued || '',
      valid: daysRemaining > 0,
      daysRemaining
    };
  } catch {
    return null;
  }
}

export default function PortalPage() {
  const [licenseKey, setLicenseKey] = useState('');
  const [licenseInfo, setLicenseInfo] = useState<LicenseInfo | null>(null);
  const [error, setError] = useState('');

  const handleValidate = () => {
    setError('');
    setLicenseInfo(null);
    
    if (!licenseKey.trim()) {
      setError('Please enter a license key');
      return;
    }
    
    const info = validateLicense(licenseKey.trim());
    if (info) {
      setLicenseInfo(info);
    } else {
      setError('Invalid license key format');
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 py-20">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-white mb-4">
            Customer Portal
          </h1>
          <p className="text-xl text-gray-400">
            Manage your Adaptive-K licenses and downloads
          </p>
        </div>

        {/* License Validation */}
        <div className="bg-gray-800 rounded-xl p-8 mb-8">
          <h2 className="text-2xl font-semibold text-white mb-6">
            Validate Your License
          </h2>
          <div className="space-y-4">
            <div>
              <label className="block text-gray-300 mb-2">License Key</label>
              <input
                type="text"
                placeholder="Enter your license key..."
                value={licenseKey}
                onChange={(e) => setLicenseKey(e.target.value)}
                className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-cyan-500"
              />
            </div>
            <button 
              onClick={handleValidate}
              className="w-full bg-cyan-600 hover:bg-cyan-500 text-white font-semibold py-3 px-6 rounded-lg transition-colors"
            >
              Validate License
            </button>
            
            {/* Error Message */}
            {error && (
              <div className="bg-red-900/50 border border-red-500 rounded-lg p-4 text-red-300">
                {error}
              </div>
            )}
            
            {/* License Info */}
            {licenseInfo && (
              <div className={`rounded-lg p-6 border ${licenseInfo.valid ? 'bg-green-900/30 border-green-500' : 'bg-red-900/30 border-red-500'}`}>
                <div className="flex items-center mb-4">
                  {licenseInfo.valid ? (
                    <svg className="w-8 h-8 text-green-400 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  ) : (
                    <svg className="w-8 h-8 text-red-400 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  )}
                  <span className={`text-xl font-semibold ${licenseInfo.valid ? 'text-green-400' : 'text-red-400'}`}>
                    {licenseInfo.valid ? 'License Valid' : 'License Expired'}
                  </span>
                </div>
                
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-400">Company:</span>
                    <p className="text-white font-medium">{licenseInfo.company}</p>
                  </div>
                  <div>
                    <span className="text-gray-400">Tier:</span>
                    <p className="text-white font-medium capitalize">{licenseInfo.tier}</p>
                  </div>
                  <div>
                    <span className="text-gray-400">Email:</span>
                    <p className="text-white font-medium">{licenseInfo.email}</p>
                  </div>
                  <div>
                    <span className="text-gray-400">Issued:</span>
                    <p className="text-white font-medium">{licenseInfo.issued}</p>
                  </div>
                  <div>
                    <span className="text-gray-400">Expires:</span>
                    <p className="text-white font-medium">{licenseInfo.expires}</p>
                  </div>
                  <div>
                    <span className="text-gray-400">Days Remaining:</span>
                    <p className={`font-medium ${licenseInfo.daysRemaining > 30 ? 'text-green-400' : licenseInfo.daysRemaining > 0 ? 'text-yellow-400' : 'text-red-400'}`}>
                      {licenseInfo.daysRemaining > 0 ? licenseInfo.daysRemaining : 'Expired'}
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Quick Actions */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          
          {/* Download SDK */}
          <div className="bg-gray-800 rounded-xl p-6">
            <div className="text-cyan-400 mb-4">
              <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">Download SDK</h3>
            <p className="text-gray-400 mb-4">Get the latest version of Adaptive-K SDK</p>
            <div className="space-y-2">
              <code className="block w-full bg-gray-900 text-center py-3 rounded-lg text-cyan-400 font-mono text-sm">
                pip install adaptive-k-routing
              </code>
              <a 
                href="https://pypi.org/project/adaptive-k-routing/"
                target="_blank"
                rel="noopener noreferrer"
                className="block w-full bg-gray-700 hover:bg-gray-600 text-center py-2 rounded-lg text-white transition-colors"
              >
                View on PyPI →
              </a>
              <a 
                href="https://github.com/VertexData/SBM-Efficient"
                target="_blank"
                rel="noopener noreferrer"
                className="block w-full bg-gray-700 hover:bg-gray-600 text-center py-2 rounded-lg text-white transition-colors"
              >
                GitHub Source →
              </a>
            </div>
          </div>

          {/* Documentation */}
          <div className="bg-gray-800 rounded-xl p-6">
            <div className="text-cyan-400 mb-4">
              <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">Documentation</h3>
            <p className="text-gray-400 mb-4">Guides, API reference, and examples</p>
            <div className="space-y-2">
              <a 
                href="/docs/INSTALLATION_GUIDE.md"
                target="_blank"
                rel="noopener noreferrer"
                className="block w-full bg-cyan-600 hover:bg-cyan-500 text-center py-2 rounded-lg text-white transition-colors font-semibold"
              >
                📖 Installation Guide
              </a>
              <a 
                href="https://github.com/VertexData/SBM-Efficient#readme"
                target="_blank"
                rel="noopener noreferrer"
                className="block w-full bg-gray-700 hover:bg-gray-600 text-center py-2 rounded-lg text-white transition-colors"
              >
                Quick Start Guide
              </a>
              <a 
                href="https://github.com/VertexData/SBM-Efficient/tree/main/sdk"
                target="_blank"
                rel="noopener noreferrer"
                className="block w-full bg-gray-700 hover:bg-gray-600 text-center py-2 rounded-lg text-white transition-colors"
              >
                SDK Documentation
              </a>
            </div>
          </div>
        </div>

        {/* Purchase Section */}
        <div className="bg-gradient-to-r from-cyan-900/50 to-purple-900/50 rounded-xl p-8 border border-cyan-500/30">
          <h2 className="text-2xl font-semibold text-white mb-4">
            Don't have a license yet?
          </h2>
          <p className="text-gray-300 mb-6">
            Get a commercial license to unlock CUDA optimizations, priority support, and enterprise features.
          </p>
          <div className="flex flex-wrap gap-4">
            <a 
              href="#professional"
              className="bg-cyan-600 hover:bg-cyan-500 text-white font-semibold py-3 px-6 rounded-lg transition-colors"
            >
              Professional - €2,500/year
            </a>
            <Link 
              href="/#contact"
              className="bg-purple-600 hover:bg-purple-500 text-white font-semibold py-3 px-6 rounded-lg transition-colors"
            >
              Enterprise - Contact Sales
            </Link>
          </div>
        </div>

        {/* Support */}
        <div className="mt-8 text-center">
          <p className="text-gray-400">
            Need help? Contact us at{' '}
            <a href="mailto:amministrazione@vertexdata.it" className="text-cyan-400 hover:underline">
              amministrazione@vertexdata.it
            </a>
          </p>
        </div>

        {/* Back to Home */}
        <div className="mt-8 text-center">
          <Link href="/" className="text-cyan-400 hover:text-cyan-300 transition-colors">
            ← Back to Home
          </Link>
        </div>
      </div>
    </div>
  );
}
