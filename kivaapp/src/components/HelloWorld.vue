<template>
  <v-container>
    <v-layout
      text-xs-center
      wrap
    >
      <v-flex>
        <!-- IMPORTANT PART! -->
<form>
          <v-text-field
            v-model="loanAmount"
            label="Loan Amount"
            required
          ></v-text-field>
          <v-text-field
            v-model="lenderTerm"
            label="Lender Term"
            required
          ></v-text-field>
          <v-text-field
            v-model="numBorrowers"
            label="Num Borrowers"
            required
          ></v-text-field>
          <v-text-field
            v-model="percentFemale"
            label="Percent Female"
            required
          ></v-text-field>
          <v-text-field
            v-model="plannedDuration"
            label="Planned Duration"
            required
          ></v-text-field>
<v-btn @click="submit">submit</v-btn>
          <v-btn @click="clear">clear</v-btn>
        </form>
<br/>
        <br/>
<h1>Predicted Class is: {{ predictedClass }}</h1>
<!-- END: IMPORTANT PART! -->
      </v-flex>
    </v-layout>
  </v-container>
</template>
<script>
  import axios from 'axios'
export default {
    name: 'HelloWorld',
    data: () => ({
      loanAmount: '',
      lenderTerm: '',
      numBorrowers: '',
      percentFemale: '',
      plannedDuration: '',
      predictedClass : ''
    }),
    methods: {
    submit () {
      axios.post('http://127.0.0.1:5000/predict', {
        LOAN_AMOUNT: this.loanAmount,
        LENDER_TERM: this.lenderTerm,
        NUM_BORROWERS: this.numBorrowers,
        PERCENT_FEMALE: this.percentFemale,
        PLANNED_DURATION: this.plannedDuration
      })
      .then((response) => {
        this.predictedClass = response.data.class
      })
    },
    clear () {
      this.loanAmount = ''
      this.lenderTerm = ''
      this.numBorrowers = ''
      this.percentFemale = ''
      this.plannedDuration = ''
    }
  }
}
</script>