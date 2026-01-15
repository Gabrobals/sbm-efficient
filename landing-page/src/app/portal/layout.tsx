import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Customer Portal - Adaptive-K | Vertex Data',
  description: 'Manage your Adaptive-K licenses, download SDK, and access documentation.',
  robots: 'noindex, nofollow', // Portal should not be indexed
};

export default function PortalLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
