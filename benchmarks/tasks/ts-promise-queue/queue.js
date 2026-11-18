// queue.js: promise queue that deadlocks when a task rejects.
class PromiseQueue {
  constructor() {
    this.queue = [];
    this.running = false;
  }
  enqueue(fn) {
    this.queue.push(fn);
    this._run();
  }
  _run() {
    if (this.running) return;
    const fn = this.queue.shift();
    if (!fn) {
      this.running = false;
      return;
    }
    this.running = true;
    // BUG: a rejected task never releases the lock, deadlocking the queue.
    Promise.resolve().then(fn).then(() => {
      this.running = false;
      this._run();
    });
  }
}
module.exports = { PromiseQueue };
