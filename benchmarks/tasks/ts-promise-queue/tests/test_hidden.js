const { PromiseQueue } = require("../queue");

(async function () {
  const q = new PromiseQueue();
  let ranSecond = false;
  q.enqueue(() => Promise.reject(new Error("boom")));
  q.enqueue(() => {
    ranSecond = true;
    return Promise.resolve();
  });
  await new Promise((r) => setTimeout(r, 60));
  if (!ranSecond) throw new Error("queue deadlocked after a rejection");
  console.log("ok");
})();
