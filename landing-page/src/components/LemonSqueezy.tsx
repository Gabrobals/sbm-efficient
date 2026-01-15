'use client';

import { useEffect } from 'react';

interface LemonSqueezyButtonProps {
  productId: string;
  variant?: 'primary' | 'secondary';
  children: React.ReactNode;
  className?: string;
}

/**
 * LemonSqueezy Checkout Button
 * 
 * Integrates with LemonSqueezy's checkout overlay.
 * Replace STORE_ID and productId with your actual values from LemonSqueezy dashboard.
 * 
 * Setup:
 * 1. Create products in LemonSqueezy dashboard
 * 2. Get the checkout URL for each product
 * 3. Update the STORE_ID constant below
 */

// TODO: Replace with your actual LemonSqueezy store ID
const STORE_ID = 'vertexdata'; // Your store subdomain

export function LemonSqueezyButton({ 
  productId, 
  variant = 'primary',
  children,
  className = ''
}: LemonSqueezyButtonProps) {
  
  useEffect(() => {
    // Load LemonSqueezy script
    const script = document.createElement('script');
    script.src = 'https://assets.lemonsqueezy.com/lemon.js';
    script.defer = true;
    document.head.appendChild(script);
    
    return () => {
      // Cleanup if needed
    };
  }, []);

  const checkoutUrl = `https://${STORE_ID}.lemonsqueezy.com/checkout/buy/${productId}`;
  
  const baseStyles = "font-semibold py-3 px-6 rounded-lg transition-colors inline-block text-center";
  const variantStyles = variant === 'primary' 
    ? "bg-cyan-600 hover:bg-cyan-500 text-white"
    : "bg-purple-600 hover:bg-purple-500 text-white";

  return (
    <a
      href={checkoutUrl}
      className={`lemonsqueezy-button ${baseStyles} ${variantStyles} ${className}`}
      data-lemon-squeezy
    >
      {children}
    </a>
  );
}

/**
 * Product IDs - Updated with actual LemonSqueezy products
 * 
 * Store: vertexdata.lemonsqueezy.com
 */
export const PRODUCTS = {
  PROFESSIONAL: '1220068', // €2,500/year - Variant ID
  ENTERPRISE: '1220176',   // €4,000/6months (€8,000/year) - Variant ID
} as const;

/**
 * Usage Example:
 * 
 * import { LemonSqueezyButton, PRODUCTS } from '@/components/LemonSqueezy';
 * 
 * <LemonSqueezyButton productId={PRODUCTS.PROFESSIONAL}>
 *   Buy Professional - €2,500/year
 * </LemonSqueezyButton>
 */
