const { updateState } = require("../state");

function testMerge() {
  const s = { a: 1, b: 2 };
  const r = updateState(s, { b: 3, c: 4 });
  if (!(r.a === 1 && r.b === 3 && r.c === 4)) {
    throw new Error("state fields were not merged");
  }
  if (r === s) throw new Error("state must not be mutated in place");
}

if (require.main === module) {
  testMerge();
  console.log("ok");
}
