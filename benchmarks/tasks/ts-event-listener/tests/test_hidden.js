const { setup } = require("../events");

function testDedup() {
  let calls = 0;
  const fn = () => {
    calls++;
  };
  const e = setup();
  e.on(fn);
  e.on(fn); // duplicate registration
  e.trigger();
  if (calls !== 1) throw new Error("expected 1 call, got " + calls);
}

if (require.main === module) {
  testDedup();
  console.log("ok");
}
