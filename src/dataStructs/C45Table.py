import pandas as pd
import linecache

class C45Table:
    def __init__(self, trainingDf = None, domain_opts = None, 
                domain_sizes = None, attributes = None, classVar = None):
        ''' Init a new C45Table table, send in a pandas dataframe. The first row should 
            have the class variables, the second row the class counts, the third row 
            will have the class column being investigated for supevised learning.
        '''
        self.trainingDf = trainingDf
        self.domain_opts = domain_opts
        self.domain_sizes = domain_sizes
        self.attributes = attributes
        self.classVar = classVar


    def __repr__(self):
        ret = str((self.trainingDf))
        ret += (f'\nCLASSVAR: {self.classVar}')
        ret += (f'\nATTRIBUTES: {self.attributes}')
        ret += (f'\nDOMAIN_OPS (SIZES):')
        for key, value in self.domain_opts.items(): 
            ret += f'\n  {key} ({self.domain_sizes[key]}): {value}'
        return ret


    def buildFromCSV(self, csvfile):
        ''' Helper function to populate a C45Table instance with data from a csv file.
            CSV INFO: - First row should have the class variables, 
                      - Second row the class counts, 
                      - Third row will have the class column being investigated for 
                        supevised learning.
            Function will print any errors to console and return -1. In this case
            the C45Table object is unknown and should not be used.
        '''
        try:
            self.domain_opts = {}
            self.domain_sizes = {}
            self.trainingDf = pd.read_csv(csvfile, skiprows=[1, 2])  # Skip info rows
            self.attributes = self.trainingDf.columns.values.tolist()
            self.classVar = linecache.getline(csvfile, 3).strip().strip('"')
            attributesHeader = linecache.getline(csvfile, 2).strip().split(',')
            if len(attributesHeader) != len(self.trainingDf.columns):
                raise BaseException('Bad attributes header')
            # Generate info dicts for each column
            col = 0
            for (columnName, columnData) in self.trainingDf.iteritems():
                # Handle -1 and 0 values in the header
                if int(attributesHeader[col]) == -1: self.attributes.remove(columnName)
                if int(attributesHeader[col]) == 0: 
                    self.domain_opts[columnName] = None
                    self.domain_sizes[columnName] = 0
                else:
                    self.domain_opts[columnName] = set(columnData)
                    self.domain_sizes[columnName] = len(self.domain_opts[columnName])
                col += 1
            self.attributes.remove(self.classVar)
        except Exception as e:
            print(f'ERR: FAILED TO BUILD FROM CSV: {e}')
            return -1
        else: return 0


    def getMostCommonClassOpt(self, attribute):
        ''' Given an attribute row, return a tuple as (most common opt, count) '''
        if attribute not in self.attributes: return None
        counts = self.trainingDf[attribute].value_counts()
        maxInCol = (counts.idxmax(), counts.max())
        return maxInCol


    def delAttribute(self, attribute):
        ''' Removes an attribute column from the current instance '''
        if attribute not in self.attributes: return False
        self.attributes.remove(attribute)
        del self.domain_opts[attribute]
        del self.domain_sizes[attribute]
        return True


    def checkIfHomogenous(self):
        ''' Returns True if the classVar is homogenous, False otherwise '''
        return self.domain_sizes[self.classVar] == 1


    def getNumericAttributes(self):
        ''' Return a list of numeric attributes '''
        ret = []
        for attribute in self.attributes:
            if self.domain_sizes[attribute] == 0: ret.append(attribute) 
        return ret


    def getCategoricalAttributes(self):
        ''' Return a list of categorical attributes '''
        ret = []
        for attribute in self.attributes:
            if self.domain_sizes[attribute] != 0: ret.append(attribute) 
        return ret

    def applyRestrictionsFile(self, restrictionsFilePath):
        pass


    def filterAndDuplicate(self, spitAttribute):
        ''' A very helpful function for implementing C45, call this before reccursion
            and a list of new C45Table objects will be created based on all of the 
            possible classes in spitAttribute. The new objects will be filtered by
            class and the attribute list reduced to represent the change.
        '''
        # Don't every try with a numeric attribute
        categoricalAttributes = self.getCategoricalAttributes()
        if spitAttribute not in categoricalAttributes: return None
        returnNewTablesList = []
        # Create a new C45Table for each possible value of the attribute
        for classOpt in self.domain_opts[spitAttribute]:
            newDomainOpts = {}
            newDomainSizes = {}
            # Copies the df, filtering based on the current class
            newDf = self.trainingDf[self.trainingDf[spitAttribute] == classOpt]
            newDf = newDf.drop(spitAttribute, axis=1)
            # Recalculate attribute lists
            newAttributes = newDf.columns.values.tolist()
            newAttributes.remove(self.classVar)
            for (columnName, columnData) in newDf.iteritems():
                # Check if column is numeric
                if columnName in categoricalAttributes:
                    newDomainOpts[columnName] = set(columnData)
                    newDomainSizes[columnName] = len(self.domain_opts[columnName])
                else:
                    newDomainOpts[columnName] = None
                    newDomainSizes[columnName] = 0
            # Create the new C45Table and append to returnlist
            newC45Table = C45Table(newDf, newDomainOpts, newDomainSizes, newAttributes, self.classVar)
            returnNewTablesList.append(newC45Table)
        return returnNewTablesList

        
# Just for demonstration
if __name__ == "__main__":
    # Set up a table
    table = C45Table()
    success = table.buildFromCSV('in/c45Input2/winequality-white-fixed.csv')
    if success == -1: exit("Table did not build properly")

    # Print a table (prints the df and attribute info)
    #print(table)

    # Other fun(ctions)
    print(table.checkIfHomogenous())
    print(table.getNumericAttributes())
    print(table.getCategoricalAttributes())
    print(table.filterAndDuplicate('Age'))
    print(table.attributes)
    print(table.domain_sizes['quality'])
    #t2 = table.filterAndDuplicate('Sex')
    #print(t2[1].attributes)