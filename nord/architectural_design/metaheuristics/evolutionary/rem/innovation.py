"""
Created on 2018-10-29

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""


class Innovation(object):

    def __init__(self):
        self.inno_number = 1
        self.current_genes = dict()

    def new_generation(self):
        self.current_genes = dict()

    def assign_number(self, gene):
        if gene not in self.current_genes:
            self.current_genes[gene] = self.inno_number
            self.inno_number += 1
        gene.innovation_number = self.current_genes[gene]
