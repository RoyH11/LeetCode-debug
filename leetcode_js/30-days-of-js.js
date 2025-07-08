// 2667

const { create, sum } = require("lodash");

/**
 * @return {Function}
 */
var createHelloWorld = function() {
    
  return function(...args) {
    return "Hello World"
  }
};

/**
 * const f = createHelloWorld();
 * f(); // "Hello World"
 */

// 2620 
/**
 * @param {number} n
 * @return {Function} counter
 */
var createCounter = function(n) {
    
  return function() {
    n = n + 1;
    return n - 1; 
  };
};

const counter = createCounter(5); 

// console.log(counter()); // 5
// console.log(counter()); // 6
// console.log(counter()); // 7


/** 
 * const counter = createCounter(10)
 * counter() // 10
 * counter() // 11
 * counter() // 12
 */

// 2704
/**
 * @param {string} val
 * @return {Object}
 */
var expect = function(val) {
  return {
    toBe: function(expected) {
      if (val !== expected) {
        throw new Error("Not Equal"); 
      }
      return true;
    }, 
    notToBe: function(expected) {
      if (val === expected) {
        throw new Error("Equal");
      }
      return true;
    }
  };
};

/**
 * expect(5).toBe(5); // true
 * expect(5).notToBe(5); // throws "Equal"
 */

/**
 * @param {string} val
 * @return {Object}
 */
var expect = (val) => ({
    
  toBe: (expected) => {
    if (val !== expected) {
      throw new Error("Not Equal"); 
    }
    return true;
  }, 
  notToBe: (expected) => {
    if (val === expected) {
      throw new Error("Equal");
    }
    return true;
  }

});

// 2665
/**
 * @param {integer} init
 * @return { increment: Function, decrement: Function, reset: Function }
 */
var createCounter = function(init) {
  let count = init;

  return {
    increment: () => {
      count += 1;
      return count;
    }, 
    decrement: () => {
      count -= 1;
      return count;
    },
    reset: () => {
      count = init;
      return count;
    }
  };
};

/**
 * const counter = createCounter(5)
 * counter.increment(); // 6
 * counter.reset(); // 5
 * counter.decrement(); // 4
 */

// 2635 
/**
 * @param {number[]} arr
 * @param {Function} fn
 * @return {number[]}
 */
var map = function(arr, fn) {
  const result = [];

  for (let i = 0; i < arr.length; i++) {
    result.push(fn(arr[i], i));
  }

  return result;
};

// 2634
/**
 * @param {number[]} arr
 * @param {Function} fn
 * @return {number[]}
 */
var filter = function(arr, fn) {
  const result = [];
  for (let i = 0; i < arr.length; i++) {
    if (fn(arr[i], i)) {
      result.push(arr[i]);
    }
  }
  return result;
};

// 2626
/**
 * @param {number[]} nums
 * @param {Function} fn
 * @param {number} init
 * @return {number}
 */
var reduce = function(nums, fn, init) {
  for (let i = 0; i < nums.length; i++ ) {
    init = fn(init, nums[i], i);
  }
  return init;
};

// 2629
/**
 * @param {Function[]} functions
 * @return {Function}
 */
var compose = function(functions) {
    
  return function(x) {
    if (functions.length === 0) {
      return x;
    }
    let result = x; 
    for (let i = functions.length - 1; i >=0; i--) {
      result = functions[i](result);
    }
    return result;
  }
};

/**
 * const fn = compose([x => x + 1, x => 2 * x])
 * fn(4) // 9
 */

//2703
/**
 * @param {...(null|boolean|number|string|Array|Object)} args
 * @return {number}
 */
var argumentsLength = function(...args) {
  return args.length;
};

/**
 * argumentsLength(1, 2, 3); // 3
 */

//2703 arrow 
/**
 * @param {...(null|boolean|number|string|Array|Object)} args
 * @return {number}
 */
var argumentsLength = (...args) => args.length;

// 2666
/**
 * @param {Function} fn
 * @return {Function}
 */
var once = function(fn) {
  let called = false;
  let result;

  return function(...args){
    if (!called) {
      called = true;
      result = fn(...args);
      return result;
    }
    return undefined; // Return undefined if fn has already been called
  }
};

/**
 * let fn = (a,b,c) => (a + b + c)
 * let onceFn = once(fn)
 *
 * onceFn(1,2,3); // 6
 * onceFn(2,3,6); // returns undefined without calling fn
 */

// 2666 arrow
/**
 * @param {Function} fn
 * @return {Function}
 */
var once = (fn) => {
  let called = false;
  let result;

  return (...args) => {
    if (!called) {
      called = true;
      result = fn(...args);
      return result;
    }
    return undefined; // Return undefined if fn has already been called
  };
}

// 2623 
/**
 * @param {Function} fn
 * @return {Function}
 */
function memoize(fn) {

  const cache = new Map();

  return function(...args) {
    const key = JSON.stringify(args);
    if (cache.has(key)) {
      return cache.get(key);
    }
    const result = fn(...args);
    cache.set(key, result);
    return result;
  }
}


/** 
 * let callCount = 0;
 * const memoizedFn = memoize(function (a, b) {
 *	 callCount += 1;
 *   return a + b;
 * })
 * memoizedFn(2, 3) // 5
 * memoizedFn(2, 3) // 5
 * console.log(callCount) // 1 
 */

//2723
/**
 * @param {Promise} promise1
 * @param {Promise} promise2
 * @return {Promise}
 */
var addTwoPromises = async function(promise1, promise2) {
  const value1 = await promise1;
  const value2 = await promise2;
  return value1 + value2;
};

/**
 * addTwoPromises(Promise.resolve(2), Promise.resolve(2))
 *   .then(console.log); // 4
 */ 

//2723
/**
 * @param {Promise} promise1
 * @param {Promise} promise2
 * @return {Promise}
 */
var addTwoPromises = async function(promise1, promise2) {
  return Promise.all([promise1, promise2])
    .then(([value1, value2]) => value1 + value2)
};

// 2621
/**
 * @param {number} millis
 * @return {Promise}
 */
async function sleep(millis) {
  return new Promise(resolve => {
    setTimeout(resolve, millis);
  })
}

/** 
 * let t = Date.now()
 * sleep(100).then(() => console.log(Date.now() - t)) // 100
 */

// 2715
/**
 * @param {Function} fn
 * @param {Array} args
 * @param {number} t
 * @return {Function}
 */
var cancellable = function(fn, args, t) {
  const timeoutId = setTimeout(() => {
    fn(...args);
  }, t);

  const cancelFn = () => {
    clearTimeout(timeoutId);
  };
  return cancelFn;
};

// const result = [];

// const fn = (x) => x * 5;
// const args = [2], t = 20, cancelTimeMs = 50;

// const start = performance.now();

// const log = (...argsArr) => {
//   const diff = Math.floor(performance.now() - start);
//   result.push({"time": diff, "returned": fn(...argsArr)});
// }
     
// const cancel = cancellable(log, args, t);

// const maxT = Math.max(t, cancelTimeMs);

// setTimeout(cancel, cancelTimeMs);

// setTimeout(() => {
//   console.log(result); // [{"time":20,"returned":10}]
// }, maxT + 15)

// console.log("Cancellable function created, waiting for execution...");

// 2725
/**
 * @param {Function} fn
 * @param {Array} args
 * @param {number} t
 * @return {Function}
 */
var cancellable = function(fn, args, t) {
  fn(...args); // Call immediately

  const intervalId = setInterval(() => {
    fn(...args);
  }, t);

  return () => clearInterval(intervalId);
};

/**
 *  const result = [];
 *
 *  const fn = (x) => x * 2;
 *  const args = [4], t = 35, cancelTimeMs = 190;
 *
 *  const start = performance.now();
 *
 *  const log = (...argsArr) => {
 *      const diff = Math.floor(performance.now() - start);
 *      result.push({"time": diff, "returned": fn(...argsArr)});
 *  }
 *       
 *  const cancel = cancellable(log, args, t);
 *
 *  setTimeout(cancel, cancelTimeMs);
 *   
 *  setTimeout(() => {
 *      console.log(result); // [
 *                           //     {"time":0,"returned":8},
 *                           //     {"time":35,"returned":8},
 *                           //     {"time":70,"returned":8},
 *                           //     {"time":105,"returned":8},
 *                           //     {"time":140,"returned":8},
 *                           //     {"time":175,"returned":8}
 *                           // ]
 *  }, cancelTimeMs + t + 15)    
 */

// 2727
/**
 * @param {Object|Array} obj
 * @return {boolean}
 */
var isEmpty = function(obj) {
  return Object.keys(obj).length === 0;
};

// 2677
/**
 * @param {Array} arr
 * @param {number} size
 * @return {Array}
 */
var chunk = function(arr, size) {
  const result = [];
  for (let i = 0; i < arr.length; i += size) {
    result.push(arr.slice(i, i + size));
  }
  return result;
};

// 2619
/**
 * @return {null|boolean|number|string|Array|Object}
 */
Array.prototype.last = function() {
  if (this.length === 0) {
    return -1;
  }
  return this[this.length - 1];
};

/**
 * const arr = [1, 2, 3];
 * arr.last(); // 3
 */

// 2724
/**
 * @param {Array} arr
 * @param {Function} fn
 * @return {Array}
 */
var sortBy = function(arr, fn) {
  return arr.slice().sort((a, b) => fn(a) - fn(b));
};

// 2695
/**
 * @param {number[]} nums
 * @return {void}
 */
var ArrayWrapper = function(nums) {
  this.nums = nums;
};

/**
 * @return {number}
 */
ArrayWrapper.prototype.valueOf = function() {
  return this.nums.reduce((sum, num) => sum + num, 0);
}

/**
 * @return {string}
 */
ArrayWrapper.prototype.toString = function() {
  return JSON.stringify(this.nums);
}

/**
 * const obj1 = new ArrayWrapper([1,2]);
 * const obj2 = new ArrayWrapper([3,4]);
 * obj1 + obj2; // 10
 * String(obj1); // "[1,2]"
 * String(obj2); // "[3,4]"
 */

// 2726
class Calculator {
    
  /** 
     * @param {number} value
     */
  constructor(value) {
    this.result = value;
  }
    
  /** 
     * @param {number} value
     * @return {Calculator}
     */
  add(value){
    this.result += value;
    return this;
  }
    
  /** 
     * @param {number} value
     * @return {Calculator}
     */
  subtract(value){
    this.result -= value;
    return this;
  }
    
  /** 
     * @param {number} value
     * @return {Calculator}
     */  
  multiply(value) {
    this.result *= value;
    return this;
  }
    
  /** 
     * @param {number} value
     * @return {Calculator}
     */
  divide(value) {
    if (value === 0) {
      throw new Error("Division by zero is not allowed");
    }
    this.result /= value;
    return this;
  }
    
  /** 
     * @param {number} value
     * @return {Calculator}
     */
  power(value) {
    this.result **= value;
    return this;
  }
    
  /** 
     * @return {number}
     */
  getResult() {
    return this.result;
  }
}

// 2694
class EventEmitter {
  constructor() {
    this.event = {};
  }

  /**
     * @param {string} eventName
     * @param {Function} callback
     * @return {Object}
     */
  subscribe(eventName, callback) {
    if (!this.event[eventName]) {
      this.event[eventName] = [];
    }
    this.event[eventName].push(callback);

    return {
      unsubscribe: () => {
        this.event[eventName] = this.event[eventName].filter(cb => cb !== callback);
        if (this.event[eventName].length === 0) {
          delete this.event[eventName];
        }
        return undefined; // Return undefined after unsubscribing
      }
    };
  }
    
  /**
     * @param {string} eventName
     * @param {Array} args
     * @return {Array}
     */
  emit(eventName, args = []) {
    if (!this.event[eventName]) return [];

    return this.event[eventName].map(callback => {
      return callback(...args);
    })
  }
}

/**
 * const emitter = new EventEmitter();
 *
 * // Subscribe to the onClick event with onClickCallback
 * function onClickCallback() { return 99 }
 * const sub = emitter.subscribe('onClick', onClickCallback);
 *
 * emitter.emit('onClick'); // [99]
 * sub.unsubscribe(); // undefined
 * emitter.emit('onClick'); // []
 */

// 2637 
/**
 * @param {Function} fn
 * @param {number} t
 * @return {Function}
 */
var timeLimit = function(fn, t) {
    
  return async function(...args) {
    const timeoutPromise = new Promise((resuolve, reject) => {
      setTimeout(() => reject("Time Limit Exceeded"), t);
    })

    return Promise.race([
      fn(...args), 
      timeoutPromise
    ])
  }
};


const limited = timeLimit((t) => new Promise(res => setTimeout(() => res('Done'), t)), 100);
// limited(150)
//   .then(console.log)
//   .catch(console.log) // "Time Limit Exceeded" at t=100ms

// limited(50)
//   .then(console.log) // undefined at t=50ms
//   .catch(console.log);

// 2631
/**
 * @param {Function} fn
 * @return {Object}
 */
Array.prototype.groupBy = function(fn) {
  const result = {}; 

  for (let i = 0; i < this.length; i++) {
    const key = fn(this[i]);
    if (!result[key]) {
      result[key] = [];
    }
    result[key].push(this[i]);
  }
  return result;
};

/**
 * [1,2,3].groupBy(String) // {"1":[1],"2":[2],"3":[3]}
 */


// 2622
var TimeLimitedCache = function() {
  this.store = new Map(); 
};

/** 
 * @param {number} key
 * @param {number} value
 * @param {number} duration time until expiration in ms
 * @return {boolean} if un-expired key already existed
 */
TimeLimitedCache.prototype.set = function(key, value, duration) {
  const existing = this.store.get(key);

  // clear previous timeout if key already exists
  if (existing) {
    clearTimeout(existing.timeoutId);
  }

  const timeoutId = setTimeout(() => {
    this.store.delete(key); // remove key after duration
  }, duration);

  this.store.set(key, {
    value: value,
    timeoutId: timeoutId
  })

  return existing !== undefined; // return true if key already existed
};

/** 
 * @param {number} key
 * @return {number} value associated with key
 */
TimeLimitedCache.prototype.get = function(key) {
  const entry = this.store.get(key);
  if (!entry) {
    return -1; // return -1 if key does not exist or has expired
  }
  return entry.value; // return the value if key exists and has not expired
};

/** 
 * @return {number} count of non-expired keys
 */
TimeLimitedCache.prototype.count = function() {
  return this.store.size; // return the size of the store which contains non-expired keys
};

/**
 * const timeLimitedCache = new TimeLimitedCache()
 * timeLimitedCache.set(1, 42, 1000); // false
 * timeLimitedCache.get(1) // 42
 * timeLimitedCache.count() // 1
 */

// 2627
/**
 * @param {Function} fn
 * @param {number} t milliseconds
 * @return {Function}
 */
var debounce = function(fn, t) {
  let timeoutId = null;

  return function(...args) {
    if (timeoutId) {
      clearTimeout(timeoutId); // clear previous timeout if it exists
    }

    timeoutId = setTimeout(() => {
      fn(...args); // call the function after t milliseconds
      timeoutId = null; // reset timeoutId after function call
    }, t);
  };
};

/**
 * const log = debounce(console.log, 100);
 * log('Hello'); // cancelled
 * log('Hello'); // cancelled
 * log('Hello'); // Logged at t=100ms
 */

// 2722 
/**
 * @param {Array} arr1
 * @param {Array} arr2
 * @return {Array}
 */
var join = function(arr1, arr2) {
  const result = [];
  const arr1Copy = arr1.slice();
  const arr2Copy = arr2.slice();
  // Sort both arrays by id
  arr1Copy.sort((a, b) => a.id - b.id);
  arr2Copy.sort((a, b) => a.id - b.id);

  while (arr1Copy.length > 0 && arr2Copy.length > 0) {
    //compare id 
    if (arr1Copy[0].id < arr2Copy[0].id) {
      result.push(arr1Copy.shift());
    } else if (arr1Copy[0].id > arr2Copy[0].id) {
      result.push(arr2Copy.shift());
    } else {
      // if ids are equal, merge the objects
      const merged = { ...arr1Copy.shift(), ...arr2Copy.shift() };
      result.push(merged);
    }
  }

  return result.concat(arr1Copy, arr2Copy); // append remaining elements
};


// 2722 better solution
/**
 * @param {Array} arr1
 * @param {Array} arr2
 * @return {Array}
 */
var join = function(arr1, arr2) {
  const map = new Map();

  for (const obj of arr1) {
    map.set(obj.id, obj);
  }

  for (const obj of arr2) {
    if (map.has(obj.id)) {
      const merged = { ...map.get(obj.id), ...obj }; // arr2 overrides arr1
      map.set(obj.id, merged);
    } else {
      map.set(obj.id, obj);
    }
  }

  return [...map.values()].sort((a, b) => a.id - b.id); // return sorted array by id
};

// 2625
/**
 * @param {Array} arr
 * @param {number} depth
 * @return {Array}
 */
var flat = function (arr, n) {
  if (n < 1) return arr; // if depth is less than 1, return original array

  const result = []; 

  for (const item of arr) {
    if (Array.isArray(item) && n >= 1) {
      // if item is an array and depth is greater than or equal to 1,
      // recursively flatten the item with depth reduced by 1
      result.push(...flat(item, n - 1));
    } else {
      // otherwise, push the item directly
      result.push(item);
    }
  }

  return result;
};

// 2625 better solution
/**
 * @param {Array} arr
 * @param {number} depth
 * @return {Array}
 */
var flat = function (arr, n) {
  if (n === 0) return arr.slice(); // if depth is 0, return a shallow copy of the array

  const result = []; 

  for (const item of arr) {
    if (Array.isArray(item) && n > 0) {
      // if item is an array and depth is greater than or equal to 1,
      // recursively flatten the item with depth reduced by 1
      result.push(...flat(item, n - 1));
    } else {
      // otherwise, push the item directly
      result.push(item);
    }
  }

  return result;
};

// 2628 stack 
/**
 * @param {Array} arr
 * @param {number} depth
 * @return {Array}
 */
var flat = function (arr, n) {
  if (n === 0) return arr.slice(); // shallow copy when no flattening needed
  
  const stack = arr.map(item => [item, n]);
  const result = [];
  
  while (stack.length > 0) {
    const [item, depth] = stack.pop();
    
    if (Array.isArray(item) && depth > 0) {
      // Add items to stack in reverse order to maintain original order
      for (let i = 0; i < item.length; i++) {
        stack.push([item[i], depth - 1]);
      }
    } else {
      result.push(item);
    }
  }
  
  return result.reverse(); // reverse to maintain original order
};