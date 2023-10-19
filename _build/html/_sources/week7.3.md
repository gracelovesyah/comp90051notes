# Additional Resource
## CNN

- Additional Resources
    - [Youtube: L13.4 Convolutional Filters and Weight-Sharing](https://www.youtube.com/watch?v=ryJ6Bna-ZNU)
    - [Neural Networks Pt. 1: Inside the Black Box](https://www.youtube.com/watch?v=CqOfi41LfDw&t=629s)
    - [Neural Networks Pt. 2: Backpropagation Main Ideas](https://www.youtube.com/watch?v=IN2XmBhILt4)
    - [Recurrent Neural Networks (RNNs), Clearly Explained!!!](https://www.youtube.com/watch?v=AsNTP8Kwu80)
    - [Long Short-Term Memory (LSTM), Clearly Explained](https://www.youtube.com/watch?v=YCzL96nL7j0&t=2s)
    - [Transformer Neural Networks, ChatGPT's foundation, Clearly Explained!!!](https://www.youtube.com/watch?v=zxQyTK8quyY)
- Article:
    -[Deep Learning: Recurrent Neural Networks](https://medium.com/deeplearningbrasilia/deep-learning-recurrent-neural-networks-f9482a24d010#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6IjdkMzM0NDk3NTA2YWNiNzRjZGVlZGFhNjYxODRkMTU1NDdmODM2OTMiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJhenAiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJhdWQiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJzdWIiOiIxMDAzNjA5OTA5NDc0NTI3NTk1MTMiLCJoZCI6InN0dWRlbnQudW5pbWVsYi5lZHUuYXUiLCJlbWFpbCI6ImppYWhlM0BzdHVkZW50LnVuaW1lbGIuZWR1LmF1IiwiZW1haWxfdmVyaWZpZWQiOnRydWUsIm5iZiI6MTY5NzUwMzEzNiwibmFtZSI6IkppYWhlIExpdSIsInBpY3R1cmUiOiJodHRwczovL2xoMy5nb29nbGV1c2VyY29udGVudC5jb20vYS9BQ2c4b2NJQ2gtYlFNcV9Sbzl6WDZKSFJwOEN4YTVMaE1vR2xlOHdWSWFHOUdsNTE9czk2LWMiLCJnaXZlbl9uYW1lIjoiSmlhaGUiLCJmYW1pbHlfbmFtZSI6IkxpdSIsImxvY2FsZSI6ImVuIiwiaWF0IjoxNjk3NTAzNDM2LCJleHAiOjE2OTc1MDcwMzYsImp0aSI6IjQzNTI3MTZlOGM5MjZhZDJmNzM4YzQxZDM2ZDcxYTJhNDhhZTlhNGMifQ.Dh2lacUeIfsnDzZQxCtMtg9hkB1FpmJOxegNehElperfTkx4v_LqWTlhrG9YA3ogdjxPVzUTuiT0qg1iIZWY8LMx55wYpWQm1g1b-UBCaCf9c9Eewg1OnFXEmKyKadLdtGI8QHaTs7vII0vEN9HvBCbYZz2-aQcmQHdsGY7lETpnjhItNtnEUfaS337dbkIk44tEJTEp6xOHej3n9PY_-al61TKl6xEtP9pQhSva262ov97E5hp_xt9r4a0UMY3CwaENcIxgSUVTFEAnYAI-DVgqI_z2S8eNEcHhP0DhqYF5cFvHUt261m7OLJOjNZnakuqRBHHicfgkS4iTH3XDbA)
```{image} ./images/cnn1.png
:alt: cnn
:class: bg-primary mb-1
:width: 800px
:align: center
```


```{image} ./images/cnn2.png
:alt: cnn
:class: bg-primary mb-1
:width: 800px
:align: center
```

Alright, let's break down the code and understand each component, starting with the initialization of the `Conv2d` layer and moving on to the sizes of the weights and biases.

---

**1. Initialization of Conv2d Layer:**
```python
conv = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5,5), stride=(1, 1))
```

Here, a 2D convolutional layer is being created using PyTorch. The arguments provided are:

- `in_channels=1`: This means that the input has one channel. Typically, grayscale images have one channel, whereas RGB images have three.

- `out_channels=8`: This means that the convolutional layer will have 8 filters or kernels. Each filter will produce its own feature map.

- `kernel_size=(5,5)`: This defines the size of each filter, which is 5x5 in this case.

- `stride=(1, 1)`: This is the step size the filter takes as it slides across the input image. Here, it moves one pixel at a time both horizontally and vertically.

**2. Size of the Convolutional Weights:**
```python
conv.weight.size()
torch.Size([8, 1, 5, 5])
```

The size of `conv.weight` reflects the parameters of the filters in the convolutional layer:

- `8`: Number of filters (from `out_channels=8`).
  
- `1`: Depth of each filter, matching the number of channels in the input (from `in_channels=1`).

- `5, 5`: Height and width of each filter (from `kernel_size=(5,5)`).

So, the weights are shaped such that there are 8 filters, each of size 1x5x5.

**3. Size of the Bias:**
```python
conv.bias.size()
torch.Size([8])
```

Every filter in the convolutional layer has an associated bias. Since there are 8 filters, there are 8 biases.

---

Now, addressing your main question:

**Why is `conv.bias.size()` = 28 when considering the equation `8*1*5*5 + 8`?**

The equation `8*1*5*5 + 8` calculates the total number of parameters in the convolutional layer.

- `8*1*5*5` computes the number of weight parameters: 8 filters each of size 1x5x5.

- `+ 8` adds the 8 bias parameters, one for each filter.

Thus, the total number of parameters in the convolutional layer is \(8 \times 1 \times 5 \times 5 + 8 = 208\), not 28. It seems there might be some confusion between the size of `conv.bias` (which is 8) and the total parameter count (which is 208). The `conv.bias.size()` is `[8]`, representing one bias term for each of the 8 filters.

### LSTM

LSTM uses different path! 

```{image} ./images/lstm1.png
:alt: cnn
:class: bg-primary mb-1
:width: 800px
:align: center
```

 ```{image} ./images/lstm2.png
:alt: cnn
:class: bg-primary mb-1
:width: 800px
:align: center
```

**input gate**
 ```{image} ./images/lstm3.png
:alt: cnn
:class: bg-primary mb-1
:width: 800px
:align: center
```

 
**output gate**
 ```{image} ./images/lstm4.png
:alt: cnn
:class: bg-primary mb-1
:width: 800px
:align: center
```
