/*
 * author:       Michael Prim
 * contact:      michael.prim@kit.edu
 * date:         2012-05-31
 * version:      1.0
 * description:  converts variables from ROOT file with simple TTree to CSV file
 *
 * arguments:    arg1 = input root filename (or wildcard for chain of files)
 *               arg2 = treename
 *               arg3 = filename for output CSV file
 *               arg4 = comma seperated list of variables that shall be written into the CSV file
 *
 * usage:       root root2CSV.C'("file.root","tree","file.csv","var1,var2,var3,var4")'
 */

#include <iostream>
#include <string>
#include <vector>

void root2CSV(char *files, char* treename, char* outfile, char* variablelist) {
	TChain *ch1 = new TChain(treename);
	ch1->Add(files);

	long no_of_entries = (long)ch1->GetEntries();

	std::vector<std::string> vars;
	std::string varlist = variablelist;
	std::string tmp = "";
	std::string::size_type found = varlist.find_first_of(",");

	int i = 0;
	while(found != std::string::npos) {
		for(i; i < found; ++i) {
			tmp += varlist[i];
		}
		vars.push_back(tmp);
		tmp = "";
		found = varlist.find_first_of(",",found+1);
		i++;
	}
	for(i; i < varlist.size(); ++i) {
		tmp += varlist[i];
	}
	vars.push_back(tmp);

	std::cout << "Converting varlist: " << varlist << std::endl;
	for(i = 0; i < vars.size(); ++i)
		std::cout << "Variable " << i << ": " << vars[i] << std::endl;

	std::ofstream fout(outfile);
	if(fout.is_open()) {
		for(i = 0; i < vars.size(); ++i) {
			if(i > 0)
				fout << ";";
			fout << vars[i];
		}
		fout << std::endl;

		for(long j=0; j<no_of_entries; j += 1) {
			ch1->GetEntry(j);

			for(i = 0; i < vars.size(); ++i) {
				if(i > 0)
					fout << ";";
				fout << ch1->GetLeaf(vars[i].c_str())->GetValue();
			}
			fout << std::endl;
		}
		fout.close();
	} else {
		std::cerr << "ERROR: Could not open file " << outfile << std::endl;
	}
	gROOT->ProcessLine(".q");
}
