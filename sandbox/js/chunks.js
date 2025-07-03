var chunk = function(arr, size) {
  const result = [];
  for (let i = 0; i < arr.length; i += size) {
    result.push(arr.slice(i, i + size));
  }
  return result;
};
// Example usage:
const arr = [1, 2, 3, 4, 5, 6, 7, 8, 9];
const size = 3;
const chunks = chunk(arr, size);
console.log(chunks); 