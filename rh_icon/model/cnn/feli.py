    def save(self):
        revision = DB.getRevision( self.id )
        revision = (revision+1)%5

        path = '%s/best_%s.%s.%d.pkl'%(Paths.Models, self.id, self.type, revision)
        path = path.lower()
        print 'revision...', revision
        print 'saving...', path
        with open(path, 'wb') as file:
            cPickle.dump((self.convLayers,
                self.mlp.hiddenLayers,
                self.mlp.logRegressionLayer,
                self.nkerns,
                self.kernelSizes,
                self.batchSize,
                self.patchSize,
                self.hiddenSizes),
                file)

        DB.finishSaveModel( self.id, revision )

