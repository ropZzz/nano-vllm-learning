#include<iostream>
#include<cmath>

int main() {
    int head_dim = 128;
    int max_position_embeddings = 1000000;
    int rope_base = 10000;

    std::vector<double> inv_freq(head_dim / 2);
    // 生成inv_freq
    for (int i = 0; i < head_dim / 2; i++) {
        inv_freq[i] = 1.0 / std::pow(rope_base, i * 2 * 1.0 / double(head_dim / 2));
    }
    //  生成 freq
    std::vector<int> t; // 生成位置索引
    std::vector<double> freq(head_dim / 2 * max_position_embeddings); // 生成频率
    for(int i =0; i<max_position_embeddings; i++) {
        t.push_back(i);
    }
    for(int i = 0; i<max_position_embeddings * head_dim; i++) {
        for(int j = 0; j<head_dim / 2; j++) {
            freq[i * head_dim / 2 + j] = t[i] * inv_freq[j];
            
    }
    // 生成sin和cos
    std::vector<double> emb(max_position_embeddings * head_dim); // 生成sin和cos的嵌入  
    for(int i = 0; i<max_position_embeddings; i++) 
    {
        int row = i / head_dim;
        int col = i % head_dim;
        int prec = i % (head_dim / 2);
        emb[row * head_dim + col] = freq[row * head_dim / 2 + prec];
        emb_sin[row * head_dim + col] = std::sin(emb[row * head_dim + col]);
        emb_cos[row * head_dim + col] = std::cos(emb[row * head_dim + col]);
    }

}