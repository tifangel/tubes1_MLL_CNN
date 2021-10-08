import copy

class BackConv():
    def __init__(self, input, kernel, loss_grad):
        self.input = input # hasil forward propagation
        self.kernel = kernel # kernel sharing parameter
        self.in_loss_grad = loss_grad # loss gradient dari layer selanjutnya
        self.out_f_update = []
        self.out_loss_grad = loss_grad

    def conv(self, input_x, loss_grad):
        w, h = len(input_x[0]), len(input_x)
        s_loss = len(loss_grad)
        size_padding, size_stride = 0, 1
        bias = 0
        new_w = w + 2 * size_padding
        new_h = h + 2 * size_padding
        # isSharing = True

        # Initialize matrix
        vw = (w - s_loss + 2 * size_padding) // size_stride + 1
        vh = (h - s_loss + 2 * size_padding) // size_stride + 1
        output = [[0 for i in range(vh)] for j in range(vw)]
        
        # Calculate convolution
        for i in range(0, new_w, size_stride):
            if i + s_loss >= new_w + 1:
                continue
            for j in range(0, new_h, size_stride):
                if j + s_loss >= new_h + 1:
                    continue
                s = bias
                for p in range(s_loss):
                    for q in range(s_loss):
                        s += input_x[i+p][j+q] * loss_grad[p][q]
                output[i//size_stride][j//size_stride] += s
        return output
        
    def fullconv(self, filter_f, loss_grad):
        f = len(filter_f)
        s_loss = len(loss_grad)
        # size_padding, size_stride = 0, 1
        # bias = 0
        # isSharing = True

        # Flip filter
        filter_flip = copy.deepcopy(filter_f)
        filter_flip.reverse()
        for i in range(f):
            filter_flip[i].reverse()
        
        # Initialize matrix
        s_output = f + s_loss - 1
        output = [[0 for i in range(s_output)] for j in range(s_output)]

        # Calculate full convolution
        for m in range(s_output): # index for output
            start_f_m = f - m - 1
            start_i = (-1 * start_f_m) if (start_f_m < 0) else 0
            last_i = (f - start_f_m) if (f - start_f_m < s_loss) else s_loss
            for n in range(s_output):
                start_f_n = f - n - 1
                s = 0
                start_j = (-1 * start_f_n) if (start_f_n < 0) else 0
                last_j = (f - start_f_n) if (f - start_f_n < s_loss) else s_loss
                for i in range(start_i, last_i): # iterasi index loss_grad
                    f_i = start_f_m + i
                    for j in range(start_j, last_j):
                        f_j = start_f_n + j
                        s += filter_flip[f_i][f_j] * loss_grad[i][j]
                output[m][n] = s
        return output
    
    def back_conv(self):
        for i in range(len(self.kernel)-1, -1, -1):
            self.out_f_update.insert(0, self.conv(self.input[i], self.out_loss_grad))
            self.out_loss_grad = self.fullconv(self.kernel[i], self.out_loss_grad)
        return self.out_f_update, self.out_loss_grad

# input = [ [ [1, 1, 2, 4],
#           [5, 6, 7, 8],
#           [3, 2, 1, 0],
#           [1, 2, -3, 4] ] ]
# kernel = [ [ [1, 1, 2, 4],
#           [5, 6, 7, 8],
#           [3, 2, 1, 0],
#           [1, 2, -3, 4] ] ]
# loss = [ [0, -1],
#            [1, 0] ]

# bc = BackConv(input,kernel,loss)
# f_update, loss_grad = bc.back_conv()
# print(f_update)
# print(loss_grad)