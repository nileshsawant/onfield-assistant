// Bundle the extension into a single dist/extension.js that VS Code loads
// at activation. `vscode` is marked external because it's supplied by the
// host, not shipped in the .vsix.
//
// Usage:
//   node esbuild.js            build once
//   node esbuild.js --watch    rebuild on save (for `F5` debug loop)
//
const esbuild = require('esbuild');

const watch = process.argv.includes('--watch');
const production = process.argv.includes('--production');

/** @type {import('esbuild').BuildOptions} */
const options = {
  entryPoints: ['src/extension.ts'],
  bundle: true,
  format: 'cjs',
  platform: 'node',
  target: 'node20',
  outfile: 'dist/extension.js',
  external: ['vscode'],
  sourcemap: !production,
  minify: production,
  logLevel: 'info'
};

async function main() {
  if (watch) {
    const ctx = await esbuild.context(options);
    await ctx.watch();
    console.log('[esbuild] watching for changes...');
  } else {
    await esbuild.build(options);
  }
}

main().catch((err) => {
  console.error('[esbuild] build failed:', err);
  process.exit(1);
});
