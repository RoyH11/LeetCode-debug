const blogs = [
  {
    _id: '5a422a851b54a676234d17f7',
    title: 'React patterns',
    author: 'Michael Chan',
    url: 'https://reactpatterns.com/',
    likes: 7,
    __v: 0
  },
  {
    _id: '5a422aa71b54a676234d17f8',
    title: 'Go To Statement Considered Harmful',
    author: 'Edsger W. Dijkstra',
    url: 'http://www.u.arizona.edu/~rubinson/copyright_violations/Go_To_Considered_Harmful.html',
    likes: 5,
    __v: 0
  },
  {
    _id: '5a422b3a1b54a676234d17f9',
    title: 'Canonical string reduction',
    author: 'Edsger W. Dijkstra',
    url: 'http://www.cs.utexas.edu/~EWD/transcriptions/EWD08xx/EWD808.html',
    likes: 12,
    __v: 0
  },
  {
    _id: '5a422b891b54a676234d17fa',
    title: 'First class tests',
    author: 'Robert C. Martin',
    url: 'http://blog.cleancoder.com/uncle-bob/2017/05/05/TestDefinitions.htmll',
    likes: 10,
    __v: 0
  },
  {
    _id: '5a422ba71b54a676234d17fb',
    title: 'TDD harms architecture',
    author: 'Robert C. Martin',
    url: 'http://blog.cleancoder.com/uncle-bob/2017/03/03/TDD-Harms-Architecture.html',
    likes: 0,
    __v: 0
  },
  {
    _id: '5a422bc61b54a676234d17fc',
    title: 'Type wars',
    author: 'Robert C. Martin',
    url: 'http://blog.cleancoder.com/uncle-bob/2016/05/01/TypeWars.html',
    likes: 2,
    __v: 0
  }
]

const listWithOneBlog = [
  {
    _id: '5a422aa71b54a676234d17f8',
    title: 'Go To Statement Considered Harmful',
    author: 'Edsger W. Dijkstra',
    url: 'https://homepages.cwi.nl/~storm/teaching/reader/Dijkstra68.pdf',
    likes: 5,
    __v: 0
  }
]

const tiedLikesBlogs = [
  {
    _id: '5a422ba71b54a676234d17fb',
    title: 'TDD harms architecture',
    author: 'Robert C. Martin',
    url: 'http://blog.cleancoder.com/uncle-bob/2017/03/03/TDD-Harms-Architecture.html',
    likes: 10,
    __v: 0
  },
  {
    _id: '5a422bc61b54a676234d17fc',
    title: 'Type wars',
    author: 'Robert C. Martin',
    url: 'http://blog.cleancoder.com/uncle-bob/2016/05/01/TypeWars.html',
    likes: 10,
    __v: 0
  }
]


// // count the number of blogs for each author
// const countBlogsByAuthor = (blogs) => {
//   return blogs.reduce((acc, blog) => {
//     acc[blog.author] = (acc[blog.author] || 0) + 1
//     return acc
//   }, {})
// }

// // call the function to count blogs by author
// const blogsByAuthorCount = countBlogsByAuthor(blogs)
// console.log(blogsByAuthorCount)


// const _ = require('lodash')
// // find the author with the most blogs with lodash
// const mostBlogs = (blogs) => {
//   const authorBlogsCount = _.countBy(blogs, 'author')
//   // return the author with the most blogs in the format { author: 'Author Name', blogs: number }
  
//   const topAuthor = _.maxBy(
//     Object.entries(authorBlogsCount),
//     ([, count]) => count
//   )

//   return {
//     author: topAuthor[0],
//     blogs: topAuthor[1]
//   }
  
// }

// console.log(mostBlogs(blogs))

const _ = require('lodash')
mostLikes = (blogs) => {
  if (blogs.length === 0) {
    return {}
  }

  const authorLikes = _.reduce(blogs, (result, blog) => {
    result[blog.author] = (result[blog.author] || 0) + blog.likes
    return result
  }, {})

  console.log(authorLikes);
  

  const topAuthor = _.maxBy(
    Object.entries(authorLikes),
    ([, likes]) => likes
  )

  return {
    author: topAuthor[0],
    likes: topAuthor[1]
  }
}

console.log(mostLikes(blogs))