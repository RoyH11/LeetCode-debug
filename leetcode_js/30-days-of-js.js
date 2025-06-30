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