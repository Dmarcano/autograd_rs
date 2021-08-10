# Introduction 

Deep learning has quickly become a mainstay of how people perceive machine learning as a whole and it has been at the forefront of much research and community effort. These days there are plenty of machine learning libraries and frameworks in a plethora of languages. But someone might still be interested in making their own, some might feel like learning by doing, that by making their own mini framework one might better understand why Pytorch or Tensorflow make the decisions they do. What you learn very quickly is that modern deep learning frameworks do not implement their methods that many neural network articles and tutorials talk about, instead they utilize what some people call "automatic differentiation". I was frankly very puzzled what this automatic differentiation was and intimidated by it but I kept coming back and eventually decided to create this book to help both myself and hopefully others in the future to better understand automatic differentiation. 


This book serves as a both a scratchpad and a learning resource through my journey of learning a bit of the theory 
and the algorithms that make up automatic differentiation. I want to share some of that learning process in the hopes that it can help
someone else who is also interested in learning AD to read another viewpoint. 

## Summary

The book is split up into a few chapters that first focus on the theory and math behind automatic differentiation. Then it goes over different types of automatic differentiation with a motivating example. 

Then I'm going to focus on implementing AD using the Rust programming language by more or less closely following the math to build the intuition of how our AD system will be built.