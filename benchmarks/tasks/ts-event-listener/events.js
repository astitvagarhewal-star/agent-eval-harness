// events.js: emitter that registers duplicate listeners.
function setup() {
  const handlers = [];
  return {
    // BUG: the same listener can be registered many times.
    on(fn) {
      handlers.push(fn);
    },
    trigger() {
      handlers.forEach((h) => h());
    },
  };
}
module.exports = { setup };
