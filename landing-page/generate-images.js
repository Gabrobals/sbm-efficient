const sharp = require('sharp');
const fs = require('fs');
const path = require('path');

async function generateImages() {
  const publicDir = path.join(__dirname, 'public');
  
  // Generate favicon (32x32)
  const iconSvg = fs.readFileSync(path.join(publicDir, 'icon.svg'));
  await sharp(iconSvg)
    .resize(32, 32)
    .png()
    .toFile(path.join(publicDir, 'favicon.ico'));
  console.log('✓ favicon.ico generated');

  // Generate icon-192 for PWA
  await sharp(iconSvg)
    .resize(192, 192)
    .png()
    .toFile(path.join(publicDir, 'icon-192.png'));
  console.log('✓ icon-192.png generated');

  // Generate icon-512 for PWA
  await sharp(iconSvg)
    .resize(512, 512)
    .png()
    .toFile(path.join(publicDir, 'icon-512.png'));
  console.log('✓ icon-512.png generated');

  // Generate OG image (1200x630)
  const ogSvg = fs.readFileSync(path.join(publicDir, 'og-image.svg'));
  await sharp(ogSvg)
    .resize(1200, 630)
    .png()
    .toFile(path.join(publicDir, 'og-image.png'));
  console.log('✓ og-image.png generated');

  console.log('\n✅ All images generated!');
}

generateImages().catch(console.error);
