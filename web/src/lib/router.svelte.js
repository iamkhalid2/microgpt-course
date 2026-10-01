// Tiny hash router: "#/" is the map, "#/ch/3" is chapter 3. Hash routing works on GitHub Pages with no server setup.
export const route = $state({ name: 'home', id: null });

function parse() {
  const [a, b] = location.hash.replace(/^#\/?/, '').split('/');
  if (a === 'ch' && b !== undefined && b !== '') { route.name = 'chapter'; route.id = b; }
  else if (a === 'glossary') { route.name = 'glossary'; route.id = null; }
  else { route.name = 'home'; route.id = null; }
}
window.addEventListener('hashchange', () => { parse(); window.scrollTo(0, 0); });
parse();

export const go = (path) => { location.hash = '#/' + path; };
export const href = (path) => '#/' + path;
