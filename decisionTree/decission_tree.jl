using CSV, DataFrames, DecisionTree, StatsBase

# Membaca data dari file CSV
data = CSV.read("contact_lens.csv", DataFrame)

# Fungsi encoding kolom kategorikal
function encode_column(col)
    dict = Dict(unique(col) .=> 0:length(unique(col)) .- 1)
    return [dict[v] for v in col], dict
end

# Encode semua fitur
encoded_data = DataFrame()
encoders = Dict()

# Encode semua kolom kecuali label
for name in names(data)[1:end-1]
    encoded, mapping = encode_column(data[!, name])
    encoded_data[!, name] = encoded
    encoders[name] = mapping
end

# Encode label (kolom terakhir)
encoded_class, class_map = encode_column(data[!, end])

# Konversi fitur ke Float64
y = Int.(encoded_class)

# Fungsi menghitung entropy
function my_entropy(y)
    n = length(y)
    counts = countmap(y)
    probs = [v / n for v in values(counts)]
    return -sum(p * log2(p) for p in probs)
end

# Entropy total untuk dataset
total_entropy = my_entropy(y)
println("Total Entropy: ", total_entropy)

# Fungsi menghitung Information Gain
function information_gain(feature_col, y)
    n = length(y)
    parent_entropy = my_entropy(y)
    values = unique(feature_col)
    weighted_entropy = 0.0

    for val in values
        idx = findall(x -> x == val, feature_col)
        subset_y = y[idx]
        weight = length(subset_y) / n
        weighted_entropy += weight * my_entropy(subset_y)
    end

    return parent_entropy - weighted_entropy
end

# Menghitung Information Gain untuk semua fitur
println("Information Gain Setiap Fitur:")
for name in names(encoded_data)
    gain = information_gain(encoded_data[!, name], y)
    println(" - ", name, ": ", round(gain, digits=4))
end

X = convert(Matrix{Float64}, Matrix(encoded_data))
# Membangun decision tree
model = build_tree(y, X, 4)

# Cetak hasil decision tree
println("Decision Tree:")
feature_names = names(encoded_data)
print_tree(model, 5, feature_names=feature_names)



