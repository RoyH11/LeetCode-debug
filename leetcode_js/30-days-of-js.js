// 2667

const { create } = require("lodash");

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

console.log(counter()); // 5
console.log(counter()); // 6
console.log(counter()); // 7


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