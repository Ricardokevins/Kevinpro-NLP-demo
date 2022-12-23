from transformers import pipeline
generator = pipeline('text-generation', model="/Users/sheshuaijie/Downloads/facebook-opt-2.7b")
generator("Hello, I'm am conscious and")
