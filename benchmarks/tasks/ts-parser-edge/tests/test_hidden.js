const { parseInts } = require("../parseints");

function testEmpty() {
  const r = parseInts("");
  if (!(Array.isArray(r) && r.length === 0)) {
    throw new Error("empty input should parse to []");
  }
}
function testNormal() {
  const r = parseInts("1,2,3");
  if (!(r[0] === 1 && r[1] === 2 && r[2] === 3)) {
    throw new Error("normal parse failed");
  }
}

if (require.main === module) {
  testEmpty();
  testNormal();
  console.log("ok");
}
