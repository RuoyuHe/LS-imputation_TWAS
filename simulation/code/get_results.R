library(data.table)
library(dplyr)

results_path = '/home/panwei/he000176/deepRIV/ImputedTraits/simulations/results/for_revision/'


calculate_percent_and_ci <- function(data) {
  p <- sum(data <= 0.05) / length(data)
  lower_bound <- round(p - 1.96 * sqrt(p * (1 - p) / length(data)), 3)
  upper_bound <- round(p + 1.96 * sqrt(p * (1 - p) / length(data)), 3)
  result <- paste0(round(p,3), " [", lower_bound, " - ", upper_bound, "]")
  return(result)
}

calculate_percent <- function(data) {
  p <- sum(data <= 0.05) / length(data)
  lower_bound <- round(p - 1.96 * sqrt(p * (1 - p) / length(data)), 3)
  upper_bound <- round(p + 1.96 * sqrt(p * (1 - p) / length(data)), 3)
  result <- paste0(round(p,3))
  return(result)
}


true_model = 'typeI'
a = fread(paste0(results_path,'delivr_',true_model,'_',1,'.txt'))
for(i in 2:100){
  tmp_path = paste0(results_path,'delivr_',true_model,'_',i,'.txt')
  if(file.exists(tmp_path)){
    a = rbind(a, fread(tmp_path))
  }
}

a = data.frame(gene_ind = a$gene_ind, twasl_obs = a$twasl_obs, pg_obs = a$pg_obs, pg_imp = a$pg_imp, pg_comp = a$pg_comp)
colnames(a) = c('gene_ind', 'TWAS-L Observed', 'DeLIVR Observed', 'DeLIVR Imputed', 'DeLIVR Complete')

result <- a %>%
  group_by(gene_ind) %>%
  summarise(across(everything(), ~ paste(sapply(list(.), calculate_percent), collapse = ", ")))

print(result)
result = as.data.frame(result) %>% arrange(`DeLIVR Complete`)
idx = which(result$`DeLIVR Complete`>=0.09)
length(idx)
idx = idx[2:25]
cor_genes = result$gene_ind[idx]

a = a %>% filter(!gene_ind %in% cor_genes)

res = sapply(a, calculate_percent_and_ci)




true_model = 'power'
b = fread(paste0(results_path,'delivr_',true_model,'_',1,'.txt'))
for(i in 2:100){
  tmp_path = paste0(results_path,'delivr_',true_model,'_',i,'.txt')
  if(file.exists(tmp_path)){
    b = rbind(b, fread(tmp_path))
  }
}

b = b %>% filter(!gene_ind %in% cor_genes)

res_power = sapply(b, calculate_percent_and_ci)
