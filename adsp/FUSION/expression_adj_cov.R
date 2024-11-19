library(data.table)
library(dplyr)

adj_cov <- function(y, cov){
  # match ids, y and cov are data.frames
  # assume ID is the first column
  data = merge(y, cov, by='ID')
  data = na.omit(data)
  data$sex = as.factor(data$sex)
  m = lm(y~., data=data[,-1]) # remove the ID column
  adj_y = data.frame(FID = rep(0,nrow(data)), IID = data$ID, pheno = scale(m$residuals))
  return(adj_y)
}


tissue = 'Brain_Hippocampus'

cov = fread(paste0("/home/panwei/shared/GTEx_v8/GTEx_Analysis_v8_expression_EUR/expression_covariates/",tissue,".v8.EUR.covariates.txt"),header = T,
                  data.table=F)
gene_exp = fread(paste0("/home/panwei/shared/GTEx_v8/GTEx_Analysis_v8_expression_EUR/expression_matrices/",tissue,".v8.EUR.normalized_expression.bed.gz"))
### Brain_Hippocampus
# cov = fread("/home/panwei/shared/GTEx_v8/GTEx_Analysis_v8_expression_EUR/expression_covariates/Brain_Hippocampus.v8.EUR.covariates.txt",header = T,
#                   data.table=F)
# gene_exp = fread("/home/panwei/shared/GTEx_v8/GTEx_Analysis_v8_expression_EUR/expression_matrices/Brain_Hippocampus.v8.EUR.normalized_expression.bed.gz")


colnames(gene_exp)[1] = 'chr'
gene_exp$chr = as.numeric(gsub("[^0-9.-]", "", gene_exp$chr))
gene_exp = na.omit(gene_exp)  
cat('dim gene_exp: ',dim(gene_exp),'\n')

cov_names = cov$ID
sample_ids = colnames(cov[,-1])
cov = as.data.frame(t(cov[,-1]))
colnames(cov) = cov_names
cov = data.frame(ID = sample_ids, cov)

gene_id = gene_exp$gene_id
gene_exp = as.data.frame(t(gene_exp[,-c(1:4)]))
colnames(gene_exp) = gene_id

for(i in 1:ncol(gene_exp)){
  cat('gene: ',i,'\n')
  gene_name = gene_id[i]
  tmp_y = adj_cov(data.frame(ID = rownames(gene_exp), y = gene_exp[,i]), cov)
  fwrite(tmp_y, paste0('/scratch.global/he000176/GTEx/expression_adj_cov/',tissue,'/',gene_name,'.txt'), sep=' ')
}



################################# protein expression #################################
library(data.table)
library(dplyr)

adj_cov <- function(y, cov){
  # match ids, y and cov are data.frames
  # assume ID is the first column
  data = merge(y, cov, by='ID')
  data = na.omit(data)
  data$sex = as.factor(data$sex)
  m = lm(y~., data=data[,-1]) # remove the ID column
  adj_y = data.frame(FID = rep(0,nrow(data)), IID = data$ID, pheno = scale(m$residuals))
  return(adj_y)
}


COV5TABLE = fread("/home/panwei/he000176/deepRIV/UKB/data/covariates_age_baseline.txt",header = T)
### proteomics
PROT = fread("/home/panwei/shared/UKBiobankIndiv/Olink/olink_data.txt")
white_unrelated_keep = fread('/home/panwei/shared/UKBiobankIndiv/WBA_plink_keep.txt',
                             header = F, data.table = F)
annot = fread('/home/panwei/he000176/deepRIV/ImputedTraits/code/adsp/olink_annotations.tsv')
# remove the X chromosome
annot = annot[annot$chr_hg19 != 'X',]

for(prot_ind in 1:nrow(annot)){
  print(prot_ind)
  cov5table = COV5TABLE[COV5TABLE$FID %in% white_unrelated_keep$V1,,drop=F]
  colnames(cov5table)[1] = 'ID'
  cov5table$sex = as.factor(cov5table$sex)
  
  prot = PROT[PROT$protein_id==prot_ind,]
  prot = prot[prot$eid %in% white_unrelated_keep$V1,] # white British
  prot = data.frame(ID=prot$eid, y=prot$result)
  
  prot_uniID = as.character(annot$uniprot_id[annot$ukb_code==prot_ind])
  prot_name = as.character(annot$gene_symbol[annot$ukb_code==prot_ind])
 
  idx = match(prot$ID, cov5table$ID)
  cat('na idx: ',sum(is.na(idx)),'\n')
  if(sum(is.na(idx))==0) cov5table = cov5table[idx,,drop=F]
  cat('cov and prot id aligned: ',all(cov5table$ID==prot$ID),'\n')
  
  tmp_y = adj_cov(prot, cov5table)
  fwrite(tmp_y, paste0('/scratch.global/he000176/UKB/prot_exp_adj_cov/',prot_name,'.txt'), sep=' ')
}



######## check for missing prot expression
library(data.table)
missing_annot = fread("/home/panwei/he000176/deepRIV/ImputedTraits/results/adsp/prot/stage1/missing_annot.csv")

results_folder = paste0("/scratch.global/he000176/UKB/prot_exp_adj_cov/")
files_in_folder = list.files(path = results_folder, pattern = "\\.txt$", full.names = FALSE)
prot_in_folder = sub("\\.txt$", "", files_in_folder)

idx = which(!(missing_annot$gene_symbol %in% prot_in_folder))

missed_exp = missing_annot[idx,]

COV5TABLE = fread("/home/panwei/he000176/deepRIV/UKB/data/covariates_age_baseline.txt",header = T)
### proteomics
PROT = fread("/home/panwei/shared/UKBiobankIndiv/Olink/olink_data.txt")
white_unrelated_keep = fread('/home/panwei/shared/UKBiobankIndiv/WBA_plink_keep.txt',
                             header = F, data.table = F)
for(prot_ind in 1:nrow(missed_exp)){
  print(prot_ind)
  cov5table = COV5TABLE[COV5TABLE$FID %in% white_unrelated_keep$V1,,drop=F]
  colnames(cov5table)[1] = 'ID'
  cov5table$sex = as.factor(cov5table$sex)
    
  prot = PROT[PROT$protein_id==missed_exp$ukb_code[prot_ind],]
  prot = prot[prot$eid %in% white_unrelated_keep$V1,] # white British
  prot = data.frame(ID=prot$eid, y=prot$result)
  
  prot_uniID = as.character(missed_exp$uniprot_id[prot_ind])
  prot_name = as.character(missed_exp$gene_symbol[prot_ind])
 
  idx = match(prot$ID, cov5table$ID)
  cat('na idx: ',sum(is.na(idx)),'\n')
  if(sum(is.na(idx))==0) cov5table = cov5table[idx,,drop=F]
  cat('cov and prot id aligned: ',all(cov5table$ID==prot$ID),'\n')
  
  tmp_y = adj_cov(prot, cov5table)
  fwrite(tmp_y, paste0('/scratch.global/he000176/UKB/prot_exp_adj_cov/',prot_name,'.txt'), sep=' ')
}