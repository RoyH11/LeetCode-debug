var animals = [
  { name: 'Fluffykins', species: 'rabbit' }, 
  { name: 'Caro', species: 'dog' },
  { name: 'Mittens', species: 'cat' },
  { name: 'Puffykins', species: 'rabbit' },
  { name: 'Rover', species: 'dog' },
  { name: 'Whiskers', species: 'cat' },
  { name: 'Fido', species: 'dog' },
  { name: 'Snowball', species: 'rabbit' },
]

// var names = []
// for (var i = 0; i < animals.length; i++) {
//   names.push(animals[i].name)
// }

// var names = animals.map(function(animal) {
//   return animal.name + ' is a ' + animal.species
// })

// arrow function
// var names = animals.map(animal => {
//   return animal.name
// })

// console.log(names)

// var dogs = []
// for (var i = 0; i < animals.length; i++) {
//   if (animals[i].species === 'dog') 
//     dogs.push(animals[i])
// }

// console.log(dogs)

// var isDog = function(animal) {
//   return animal.species === 'dog'
// }

// var dogs = animals.filter(isDog)
// var otherAnimals = animals.filter(animal => !isDog(animal))

// console.log(dogs)
// console.log(otherAnimals)


var orders = [
  { amount: 250 }, 
  { amount: 50 }, 
  { amount: 100 }, 
  { amount: 200 }, 
  { amount: 300 }, 
  { amount: 150 }, 
  { amount: 400 }, 
  { amount: 350 },
]

// var totalAmount = 0
// for (var i = 0; i < orders.length; i++) {
//   totalAmount += orders[i].amount
// }

// var totalAmount = orders.reduce(function(sum, order) {
//   // console.log('hello', sum, order)
//   return sum + order.amount
// }, 0)

// arrow function
var totalAmount = orders.reduce((sum, order) => sum + order.amount, 0)
console.log(totalAmount)